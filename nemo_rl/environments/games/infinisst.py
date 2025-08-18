import os
import copy
import random
import re
from typing import Any, List, Optional, TypedDict

import json
import tempfile

import ray
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoConfig, MistralForCausalLM
from safetensors.torch import load_file
import time
from tqdm import tqdm

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, RayVirtualCluster
from nemo_rl.environments.games.metricx24.predict import get_dataset
from nemo_rl.environments.games.mqm.utils import MQMExampleGenerator, build_prompt, lang_lookup, parse_mqm_answer, validate_output

class InfiniSSTConfig(TypedDict):
    scoring_model_path: str
    scoring_model_type: str
    batch_size: int
    max_turns: int
    src_lang: str
    tgt_lang: str

class InfiniSSTMetadata(TypedDict):
    features: torch.Tensor
    step: int
    max_steps: int
    src_segments: list[str]
    tgt_segments: list[str]
    chunk_frame_size: int

SENT_SPLITTERS = {
    "en": '.,!?',
    "ru": '.,!?',
    "zh": '。，！？',
}

CHAR_LANGS = set(['zh', 'ja'])
WORD_LANGS = set(['en', 'de', 'es', 'ru'])

CODE2LANG = {
    'en': 'English',
    'zh': 'Chinese',
    'ru': 'Russian',
    'de': 'German',
    'es': 'Spanish',
    'ja': 'Japanese',
}

SYSTEM_PROMPT = "You are a professional translation evaluator."
TEMPLATE = """Your task is to assess whether a translation segment successfully conveys the semantic content of the original speech according to the following criteria:

1. Key Information Recognition: Identify whether the key information in the source (e.g., proper nouns, keywords, terminologies, or sentence structures) is present in the translation.
2. Correctness Assessment: Determine whether the translation accurately conveys the speaker’s intention, without misinterpretation or contextual errors.
3. Expressiveness Assessment: Evaluate whether the translation is fluent, clear, and intuitive to human readers. It should avoid unnecessary verbosity, ambiguous phrases, or awkward grammar.

Given a source sentence and its translation, answer "Yes" if the translation meets all three criteria and answer "No" otherwise. Only output the answer, no other text.

<begin_of_source>
{}
<end_of_source>

<begin_of_translation>
{}
<end_of_translation>
"""
class LCME:
    def __init__(self, cfg: InfiniSSTConfig):
        from nemo_rl.environments.games.lcme.wmtAlign import load_alternative_model

        self.cfg = cfg
        self.tokenizer, self.model = load_alternative_model('cuda', 'BAAI/bge-m3')

    def segment(self, data: list[dict[str, str]]) -> list[str]:
        from nemo_rl.environments.games.lcme.wmtAlign import generate_overlap_and_embedding, run_vecalign_explore
        src_tgt_alignmentss = []
        features_to_overlap_emb = {}
                
        # Set batch size for embedding generation
        embedding_batch_size = 4
        
        # First pass: collect unique source texts that need embedding
        src_texts_to_generate = []
        src_features_ids = []
        features_id_to_src_text = {}
        
        for idx, instance in enumerate(data):
            features_id, doc_id = instance["id"].split('_')
            src_sentences = instance["src_sents"]
            src_text = "\n".join(src_sentences)
            
            if features_id not in features_to_overlap_emb:
                if features_id not in features_id_to_src_text:
                    features_id_to_src_text[features_id] = src_text
                    src_texts_to_generate.append(src_text)
                    src_features_ids.append(features_id)
        
        # Batch generate source embeddings in chunks
        if src_texts_to_generate:
            all_src_overlaps = []
            all_src_embeds = []
            
            for i in tqdm(range(0, len(src_texts_to_generate), embedding_batch_size), desc="Generating source embeddings"):
                batch_texts = src_texts_to_generate[i:i + embedding_batch_size]
                src_overlaps, src_embeds = generate_overlap_and_embedding(batch_texts, self.model, self.tokenizer, 10)
                all_src_overlaps.extend(src_overlaps)
                all_src_embeds.extend(src_embeds)
            
            for i, features_id in enumerate(src_features_ids):
                features_to_overlap_emb[features_id] = (all_src_overlaps[i], all_src_embeds[i])
        
        # Second pass: collect target texts that need embedding (non-empty)
        tgt_texts_to_generate = []        
        for idx, instance in enumerate(data):
            tgt_sentences = instance["tgt_sents"]
            tgt_text = "\n".join(tgt_sentences)
            tgt_texts_to_generate.append(tgt_text)
        
        # Batch generate target embeddings in chunks
        if tgt_texts_to_generate:
            all_tgt_overlaps = []
            all_tgt_embeds = []
            
            for i in tqdm(range(0, len(tgt_texts_to_generate), embedding_batch_size), desc="Generating target embeddings"):
                batch_texts = tgt_texts_to_generate[i:i + embedding_batch_size]
                tgt_overlaps, tgt_embeds = generate_overlap_and_embedding(batch_texts, self.model, self.tokenizer, 10)
                all_tgt_overlaps.extend(tgt_overlaps)
                all_tgt_embeds.extend(tgt_embeds)
            
        # Third pass: run vecalign for each instance
        pbar = tqdm(data, desc="Running VecAlign")
        for idx, instance in enumerate(pbar):
            tgt_sentences = instance["tgt_sents"]
            src_sentences = instance["src_sents"]
            ref_sentences = instance["ref_sents"]
            features_id, doc_id = instance["id"].split('_')
            
            # Get embeddings from cache/batch results
            src_overlap, src_embed = features_to_overlap_emb[features_id]
            tgt_overlap, tgt_embed = all_tgt_overlaps[idx], all_tgt_embeds[idx]

            # Time alignment
            src_tgt_alignments = run_vecalign_explore(
                "\n".join(src_sentences), "\n".join(tgt_sentences),
                src_overlap, tgt_overlap, src_embed, tgt_embed,
                doc_id, 10
            )

            src_tgt_alignmentss.append(src_tgt_alignments)
        
        return src_tgt_alignmentss

class RewardModel:
    def __init__(self, model_dir) -> None:
        config = AutoConfig.from_pretrained(model_dir)
        # config._attn_implementation = "flash_attention_2"
        self.device = torch.device('cuda')
        self.model = MistralForCausalLM(config)
        self.model.lm_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        state_dict = load_file(f"{model_dir}/model.safetensors")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(dtype=torch.bfloat16)
        self.model.to(device=self.device)
        self.model.eval()

    @torch.no_grad()
    def _score(self, prompts, chosens) -> List[float]:
        # Concat prompt and chosen, append eos_id
        input_ids_list = [self.tokenizer.encode(prompt) + self.tokenizer.encode(chosen) + [self.tokenizer.eos_token_id] for prompt, chosen in zip(prompts, chosens)]

        # Pad sequences to the maximum length
        max_length = max(len(ids) for ids in input_ids_list)
        padded_input_ids = [ids + [self.tokenizer.pad_token_id or self.tokenizer.eos_token_id] * (max_length - len(ids)) for ids in input_ids_list]

        # Forward pass
        input_ids = torch.tensor(padded_input_ids).to(device=self.device)
        logits = self.model(input_ids).logits

        # Extract logits corresponding to eos_token_id positions
        scores = []
        for i, input_ids in enumerate(input_ids_list):
            eos_position = input_ids.index(self.tokenizer.eos_token_id)
            eos_logit = logits[i, eos_position, :].squeeze().item()
            scores.append(eos_logit)

        return scores

    def score(self, sources, hypotheses, src_lang, tgt_lang) -> List[float]:
        prompts = [
            f"Translate the following {CODE2LANG[src_lang]} sentence into {CODE2LANG[tgt_lang]}:\n{source} <{tgt_lang}>"
            for source in sources
        ]
        return self._score(prompts, hypotheses)

@ray.remote
class InfiniSSTScorer:
    def __init__(self, cfg: InfiniSSTConfig):
        self.cfg = cfg
        self.sent_splitter = SENT_SPLITTERS[cfg["tgt_lang"]]
        self.segmenter = LCME(cfg)

        if 'comet' in cfg["scoring_model_type"].lower():
            from comet import download_model, load_from_checkpoint
            model_path = download_model(cfg["scoring_model_type"], saving_directory=cfg["scoring_model_path"])
            self.scoring_model = load_from_checkpoint(model_path)
            self.worst_score = 0
        elif 'metricx' in cfg["scoring_model_type"].lower():
            from nemo_rl.environments.games.metricx24 import models
            self.scoring_tokenizer = AutoTokenizer.from_pretrained(cfg["scoring_tokenizer_path"])
            self.scoring_model = models.MT5ForRegression.from_pretrained(cfg["scoring_model_path"], torch_dtype="auto")
            self.scoring_model.to("cuda")
            self.scoring_model.eval()
            self.worst_score = -25
        elif 'vip' in cfg["scoring_model_type"].lower():
            from vllm import LLM, SamplingParams
            self.sampling_params = SamplingParams(
                temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=1024, n=self.cfg["scoring_model_samples"],
            )
            self.llm = LLM(
                model=cfg["scoring_model_path"],
                max_model_len=1024,
                gpu_memory_utilization=0.80,
                enforce_eager=True,
            )
            self.worst_score = 0.0
        elif 'seed-x-rm' in cfg["scoring_model_type"].lower():
            self.reward_model = RewardModel(cfg["scoring_model_path"])
            self.worst_score = -10
        elif 'mqm' in cfg["scoring_model_type"].lower():
            from vllm import LLM, SamplingParams
            self.mqm_examples = MQMExampleGenerator(
                filepath=cfg["scoring_examples_path"],
                n=3,
                span_type="none",
            )
            self.mqm_tokenizer = AutoTokenizer.from_pretrained(cfg["scoring_model_path"])
            self.mqm_sampling_params = SamplingParams(
                temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=1024, n=self.cfg["scoring_model_samples"],
            )
            self.mqm_llm = LLM(
                model=cfg["scoring_model_path"],
                max_model_len=4096,
                gpu_memory_utilization=0.80,
                enforce_eager=True,
            )
            self.worst_score = -25
        else:
            raise ValueError(f"Invalid scoring model type: {cfg['scoring_model_type']}")
            
        self.batch_size = cfg["batch_size"]

    def predict(self, data: list[dict[str, str]]) -> list[float]:
        for instance in data:
            tgt_text = instance["tgt_text"]
            delays = instance["delays"]

            tgt_sentences = []
            tgt_delays = []
            current_sentence = ""

            paragraphs = tgt_text.split('\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    sentences = []
                    current_sentence = ""
                    for char in paragraph:
                        current_sentence += char
                        if char in self.sent_splitter:
                            if current_sentence.strip():
                                current_sentence = current_sentence.strip()
                                sentences.append(current_sentence)
                                units = current_sentence.split(' ') if self.cfg["tgt_lang"] in WORD_LANGS else list(current_sentence)
                                tgt_delays.append(delays[:len(units)])
                                delays = delays[len(units):]
                            current_sentence = ""
                    if current_sentence.strip():
                        current_sentence = current_sentence.strip()
                        sentences.append(current_sentence)
                        units = current_sentence.split(' ') if self.cfg["tgt_lang"] in WORD_LANGS else list(current_sentence)
                        tgt_delays.append(delays[:len(units)])
                        delays = delays[len(units):]
                    tgt_sentences.extend(sentences)

            instance["tgt_sents"] = tgt_sentences
            instance["tgt_delays"] = tgt_delays

        src_tgt_alignmentss = self.segmenter.segment(data)

        src_sep = '' if self.cfg["src_lang"] in ['zh', 'ja'] else ' '
        tgt_sep = '' if self.cfg["tgt_lang"] in ['zh', 'ja'] else ' '

        instance2data = []
        scorer_data = []
        latency_data = []
        latencies = [[] for _ in range(len(data))]
        quality_scores = [[] for _ in range(len(data))]
        for idx, src_tgt_alignments in enumerate(src_tgt_alignmentss):
            src_sentences = data[idx]["src_sents"]
            src_info = data[idx]["src_info"]
            ref_sentences = data[idx]["ref_sents"]
            tgt_sentences = data[idx]["tgt_sents"]
            tgt_delays = data[idx]["tgt_delays"]

            for src_indices, tgt_indices in src_tgt_alignments:
                if len(src_indices) == 0 or len(tgt_indices) == 0:
                    latencies[idx].append(self.cfg["max_latency"])
                    quality_scores[idx].append(self.worst_score)
                    continue
                src_sentence = src_sep.join([src_sentences[i] for i in src_indices])
                ref_sentence = tgt_sep.join([ref_sentences[i] for i in src_indices])
                ref_len = sum([len(ref_sentences[i].split(' ')) if self.cfg["tgt_lang"] in WORD_LANGS else len(ref_sentences[i]) for i in src_indices])
                tgt_sentence = tgt_sep.join([tgt_sentences[i] for i in tgt_indices])
                tgt_delay = [delay for i in tgt_indices for delay in tgt_delays[i]]

                latency_data.append({
                    "src_start": src_info[src_indices[0]]['start'],
                    "src_end": src_info[src_indices[-1]]['end'],
                    "ref_len": ref_len,
                    "delays": tgt_delay,
                })

                scorer_data.append({
                    "src": src_sentence,
                    "ref": ref_sentence,
                    "mt": tgt_sentence,
                })
                instance2data.append(idx)
        
        for i, latency_datum in enumerate(latency_data):
            start = latency_datum["src_start"]
            end = latency_datum["src_end"]
            ref_len = latency_datum["ref_len"]
            delays = latency_datum["delays"]

            step = (end - start) / max(ref_len, len(delays))
            latency = 0
            for j in range(len(delays)):
                latency += delays[j] - step * (j + 1) - start
            latency /= len(delays)
            latencies[instance2data[i]].append(latency)
        mean_latencies = [sum(latency_list) / len(latency_list) for latency_list in latencies]

        if 'comet' in self.cfg["scoring_model_type"].lower():
            scoring_model_scores = self.scoring_model.predict(scorer_data, batch_size=self.batch_size, gpus=1).scores
        elif 'metricx' in self.cfg["scoring_model_type"].lower():
            with tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".txt") as temp_in:
                for scorer_datum in scorer_data:
                    temp_in.write(json.dumps({
                        "source": scorer_datum["src"],
                        "reference": scorer_datum["ref"] if not self.cfg["qe"] else "",
                        "hypothesis": scorer_datum["mt"],
                    }) + '\n')
                temp_in.flush()
                input_filename = temp_in.name
            dataset = get_dataset(input_filename, self.scoring_tokenizer, 1536, "cuda", self.cfg["qe"])
            training_args = transformers.TrainingArguments(
                output_dir=os.path.dirname(input_filename),
                per_device_eval_batch_size=1,
                dataloader_pin_memory=False,
                report_to="none",
            )
            trainer = transformers.Trainer(
                model=self.scoring_model,
                args=training_args,
            )
            scoring_model_scores, _, _ = trainer.predict(test_dataset=dataset["test"])
            scoring_model_scores = [-float(score) for score in scoring_model_scores]
        elif 'vip' in self.cfg["scoring_model_type"].lower():
            scoring_model_scores = []
            messages = []
            for scorer_datum in scorer_data:
                prompt = TEMPLATE.format(scorer_datum["src"], scorer_datum["mt"])
                message = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                messages.append(message)
            outputs = self.llm.chat(
                messages,
                sampling_params=self.sampling_params,
                chat_template_kwargs={"enable_thinking": True},
            )
            for output in outputs:
                n_yes = 0
                n_no = 0
                n_other = 0
                for o in output.outputs:
                    think_end_pos = o.text.find("</think>")
                    if think_end_pos == -1:
                        continue
                    answer = o.text[think_end_pos + len("</think>"):].strip()
                    n_yes += "Yes" in answer
                    n_no += "No" in answer
                    n_other += "Yes" not in answer and "No" not in answer
                scoring_model_scores.append(float(n_yes > n_no))
        elif 'seed-x-rm' in self.cfg["scoring_model_type"].lower():
            sources = [scorer_datum["src"] for scorer_datum in scorer_data]
            hypotheses = [scorer_datum["mt"] for scorer_datum in scorer_data]
            scoring_model_scores = self.reward_model.score(
                sources, 
                hypotheses, 
                self.cfg["src_lang"], 
                self.cfg["tgt_lang"]
            )
        elif 'mqm' in self.cfg["scoring_model_type"].lower():
            messages = []
            for scorer_datum in scorer_data:
                message = build_prompt(
                    self.mqm_examples,
                    lang_lookup[self.cfg["src_lang"]],
                    lang_lookup[self.cfg["tgt_lang"]],
                    scorer_datum["src"],
                    scorer_datum["mt"],
                    scorer_datum["ref"],
                )
                messages.append(message)
            outputs = self.mqm_llm.chat(
                messages,
                sampling_params=self.mqm_sampling_params,
            )
            scoring_model_scores = []
            for output in outputs:
                scores = []
                for o in output.outputs:
                    try:
                        if validate_output(o.text):
                            scores.append(parse_mqm_answer(o.text))
                    except Exception as e:
                        pass
                scoring_model_scores.append(sum(scores) / len(scores) if len(scores) > 0 else self.worst_score)
        else:
            raise ValueError(f"Invalid scoring model type: {self.cfg['scoring_model_type']}")

        for i, idx in enumerate(instance2data):
            quality_scores[idx].append(scoring_model_scores[i])
        
        mean_quality_scores = []
        for score_list in quality_scores:
            mean_quality_scores.append(sum(score_list) / len(score_list))

        scores = [
            {
                self.cfg["scoring_model_type"]: quality_score,
                "latency": latency,
            }
            for quality_score, latency in zip(mean_quality_scores, mean_latencies)
        ]
        return scores

@ray.remote
class InfiniSSTEnv(EnvironmentInterface):
    """InfiniSST environment (Ray Actor)."""

    def __init__(self, cfg: Optional[InfiniSSTConfig] = None):
        self.cfg = cfg
        self.virtual_cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[cfg["num_gpus"]],
            use_gpus=True,
            name="infinisst_vc",
        )
        placement_groups = self.virtual_cluster.get_placement_groups()
        
        self.workers = []
        print(f"Creating {cfg['num_gpus']} workers sequentially to avoid environment installation conflicts...")
        for i in range(cfg["num_gpus"]):
            print(f"Creating worker {i+1}/{cfg['num_gpus']}...")
            pg_index = i % len(placement_groups)
            pg = placement_groups[pg_index]
            worker = InfiniSSTScorer.options(
                num_gpus=1,
                scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=pg
                )
            ).remote(cfg)
            self.workers.append(worker)
            
            # Wait for the worker to be fully initialized before creating the next one
            # This ensures sequential environment installation
            try:
                ray.get(worker.__ray_ready__.remote(), timeout=300)  # 5 minute timeout
                print(f"Worker {i+1} initialized successfully")
            except Exception as e:
                print(f"Warning: Could not verify worker {i+1} initialization: {e}")
        print("All workers created successfully")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
        self.max_turns = cfg["max_turns"]

    def compute_reward(self, message_log_batch: list[LLMMessageLogType], metadata_batch: list[InfiniSSTMetadata]) -> float:
        scorer_data = []
        features_ids = []
        for idx, (message_log, metadata) in enumerate(zip(message_log_batch, metadata_batch)):
            translation = ''.join([msg["content"] for msg in message_log if msg["role"] == "assistant"])

            delays = []
            n_chunks = 0
            for msg in message_log:
                n_chunks += int(msg['role'] == 'user')
                if msg['role'] == 'assistant' and msg['content'] != '':
                    units = msg['content'].split(' ') if self.cfg["tgt_lang"] in WORD_LANGS else list(msg['content'])
                    delays.extend([n_chunks * self.cfg["step_size"]] * len(units))

            features_id = str(abs(hash(f"{message_log[0]['features'][0]}-{message_log[0]['features'][1]}")))
            features_ids.append(features_id)
            scorer_data.append({
                "id": f"{features_id}_{idx}",
                "src_sents": metadata["src_segments"],
                "src_info": metadata["segment_info"],
                "ref_sents": metadata["tgt_segments"],
                "tgt_text": translation,
                "delays": delays,
            })
        n_worker = len(self.workers)
        scorer_data_per_worker = [scorer_data[i::n_worker] for i in range(n_worker)]
        results = ray.get([self.workers[i].predict.remote(scorer_data_per_worker[i]) for i in range(n_worker)])
        scores = []
        for i in range(len(message_log_batch)):
            scores.append(results[i % n_worker][i // n_worker])
        keys = list(scores[0].keys())
        metrics = {
            key: [score[key] for score in scores] for key in keys
        }

        quality_scores = np.array(metrics[self.cfg["scoring_model_type"]])
        latencies = np.array(metrics["latency"])
        features_ids = np.array(features_ids)

        if self.cfg["normalize"]:
            for features_id in set(features_ids):
                mask = features_ids == features_id
                mean_quality_scores = quality_scores[mask].mean()
                std_quality_scores = quality_scores[mask].std()
                quality_scores[mask] = quality_scores[mask] - mean_quality_scores
                if std_quality_scores > 0:
                    quality_scores[mask] = quality_scores[mask] / std_quality_scores

                mean_latencies = latencies[mask].mean()
                std_latencies = latencies[mask].std()
                latencies[mask] = latencies[mask] - mean_latencies
                if std_latencies > 0:
                    latencies[mask] = latencies[mask] / std_latencies

        rewards = self.cfg["alpha"] * quality_scores - self.cfg["beta"] * latencies

        return rewards, metrics

    def step(
        self, message_log_batch: list[LLMMessageLogType], metadata_batch: list[InfiniSSTMetadata]
    ) -> EnvironmentReturn:
        start_time = time.time()
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []
        all_next_metadata = []

        start_token_id = self.tokenizer.encode("<|im_start|>")[0]
        for metadata in metadata_batch:
            chunk_frame_size = metadata["chunk_frame_size"]
            content = "<|video_pad|>" * chunk_frame_size
            content = self.tokenizer.apply_chat_template( 
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            if sum(token_id == start_token_id for token_id in content) == 3:
                content = content[20:] # remove system prompt from qwen2.5
            content = self.tokenizer.decode(content)
            observations.append({"role": "user", "content": content})

            all_stop_strings.append(None)
            metadata["step"] += 1
            all_next_metadata.append(metadata)
            
        if metadata_batch[0]['step'] == self.max_turns:
            rewards, metrics = self.compute_reward(message_log_batch, metadata_batch)
            terminateds = [True] * len(message_log_batch)
        else:
            rewards, metrics = [0] * len(message_log_batch), {}
            terminateds = [False] * len(message_log_batch)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"InfiniSSTEnv.step took {elapsed:.4f} seconds")

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            metrics=metrics,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(self, batch: BatchedDataDict[Any]) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        pass