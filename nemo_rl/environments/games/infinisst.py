import os
import copy
import random
import re
from typing import Any, Optional, TypedDict

import json
import tempfile

import ray
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
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

class InfiniSSTConfig(TypedDict):
    scoring_model_path: str
    scoring_model_type: str
    batch_size: int
    granularity: str
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

class mWERAlign:
    def __init__(self, cfg: InfiniSSTConfig):
        self.cfg = cfg
        self.tgt_lang = cfg["tgt_lang"]

    def segment(self, data: list[dict[str, str]]) -> list[list[tuple[list[int], list[int]]]]:
        src_tgt_alignmentss = []
        for instance in data:
            hyp = instance["tgt_text"]
            ref = instance["ref_sents"]
            # Write hyp to a temporary file
            import tempfile
            import subprocess

            hyp_fd, hyp_path = tempfile.mkstemp(suffix=".txt", text=True)
            ref_fd, ref_path = tempfile.mkstemp(suffix=".txt", text=True)
            try:
                with open(hyp_path, "w", encoding="utf-8") as f_hyp:
                    f_hyp.write(hyp)
                with open(ref_path, "w", encoding="utf-8") as f_ref:
                    if isinstance(ref, list):
                        f_ref.write("\n".join(ref))
                    else:
                        f_ref.write(ref)
                # You can now use hyp_path and ref_path as needed
                # (e.g., pass to external tools)
                

            # Call mweralign with the specified arguments
            # -r ref.txt -t hyp.txt -o aligned.txt -m cj -l zh
                aligned_fd, aligned_path = tempfile.mkstemp(suffix=".txt", text=True)
                os.close(aligned_fd)  # We'll just use the path

                subprocess.run(
                    [
                        "mweralign",
                        "-r", ref_path,
                        "-t", hyp_path,
                        "-o", aligned_path,
                        "-l", self.tgt_lang
                    ] + (["-m", "cj"] if self.tgt_lang in CHAR_LANGS else []),
                    check=True
                )

                # Optionally, read the aligned output if needed
                with open(aligned_path, "r", encoding="utf-8") as f_aligned:
                    aligned_content = f_aligned.read()
                # You can process aligned_content as needed
                hyp_segments = aligned_content.split('\n')[:len(ref)]

                src_tgt_alignmentss.append(hyp_segments)

            finally:
                os.close(hyp_fd)
                os.close(ref_fd)
                os.remove(hyp_path)
                os.remove(ref_path)
                os.remove(aligned_path)

        return src_tgt_alignmentss


@ray.remote
class InfiniSSTScorer:
    def __init__(self, cfg: InfiniSSTConfig):
        self.cfg = cfg
        self.sent_splitter = SENT_SPLITTERS[cfg["tgt_lang"]]

        self.segmenter = mWERAlign(cfg)

        if 'comet' in cfg["scoring_model_type"].lower():
            from comet import download_model, load_from_checkpoint
            model_path = download_model(cfg["scoring_model_type"], saving_directory=cfg["scoring_model_path"])
            self.scoring_model = load_from_checkpoint(model_path)
            self.worst_score = 0
        else:
            from nemo_rl.environments.games.metricx24 import models
            self.scoring_tokenizer = AutoTokenizer.from_pretrained(cfg["scoring_tokenizer_path"])
            self.scoring_model = models.MT5ForRegression.from_pretrained(cfg["scoring_model_path"], torch_dtype="auto")
            self.scoring_model.to("cuda")
            self.scoring_model.eval()
            self.worst_score = -25
            
        self.batch_size = cfg["batch_size"]
        self.granularity = cfg["granularity"]

    def predict(self, data: list[dict[str, str]]) -> list[float]:
        breakpoint()
        src_tgt_alignmentss = self.segmenter.segment(data)

        instance2data = []
        scorer_data = []
        latency_data = []
        latencies = [[] for _ in range(len(data))]
        quality_scores = [[] for _ in range(len(data))]
        for idx, src_tgt_alignments in enumerate(src_tgt_alignmentss):
            src_sentences = data[idx]["src_sents"]
            src_info = data[idx]["src_info"]
            ref_sentences = data[idx]["ref_sents"]

            delays = data[idx]["delays"]
            segment_delays = []
            for tgt_segment in src_tgt_alignments:
                units = tgt_segment.split(' ') if self.cfg["tgt_lang"] in WORD_LANGS else list(tgt_segment)
                segment_delays.append(delays[:len(units)])
                delays = delays[len(units):]

            for segment_idx, (tgt_segment, segment_delay) in enumerate(zip(src_tgt_alignments, segment_delays)):
                if tgt_segment.strip() == "":
                    latencies[idx].append(self.cfg["max_latency"])
                    quality_scores[idx].append(self.worst_score)
                    continue
                src_sentence = src_sentences[segment_idx]
                ref_sentence = ref_sentences[segment_idx]
                ref_len = len(ref_sentence.split(' ')) if self.cfg["tgt_lang"] in WORD_LANGS else len(ref_sentence)

                latency_data.append({
                    "src_start": src_info[segment_idx]['start'],
                    "src_end": src_info[segment_idx]['end'],
                    "ref_len": ref_len,
                    "delays": segment_delay,
                })

                scorer_data.append({
                    "src": src_sentence,
                    "ref": ref_sentence,
                    "mt": tgt_segment,
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
        else:
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

        for metadata in metadata_batch:
            chunk_frame_size = metadata["chunk_frame_size"]
            content = "<|video_pad|>" * chunk_frame_size
            content = self.tokenizer.decode(
                self.tokenizer.apply_chat_template( 
                    [{"role": "user", "content": content}],
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )[20:], # remove system prompt from qwen2.5
            )
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