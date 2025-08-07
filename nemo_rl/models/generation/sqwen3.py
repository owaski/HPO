# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature, PreTrainedTokenizer, PretrainedConfig, AutoTokenizer

from vllm.config import VllmConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import (AudioProcessorItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings, WeightsMapper)

CHUNK_SIZE = 14 # this is the size of a chunk of cache aware fastconformer
MAX_N_CHUNKS = 60
PLACEHOLDER_TOKEN = "<|video_pad|>"
AUDIO_TOKEN_ID = 151656

class SQwen3ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {
            "audio": MAX_N_CHUNKS,
        }

class SQwen3Processor:
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

    def __call__(
        self,
        text=None,
        audio=None,
        return_tensors=None,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text] # type: ignore
        if audios is None:
            audios = []
        if not isinstance(audios, list):
            audios = [audios] # type: ignore

        text_inputs = self.tokenizer(text)
        audio_embeds = torch.stack(audios, dim=0)

        return BatchFeature(
            data={
                **text_inputs,
                "audio_embeds": audio_embeds,
            },
            tensor_type=return_tensors
        )

class SQwen3DummyInputsBuilder(
        BaseDummyInputsBuilder[SQwen3ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_chunks = mm_counts.get("audio", 0)
        return num_chunks * CHUNK_SIZE * PLACEHOLDER_TOKEN

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_chunks = mm_counts.get("audio", 0)
        dummy_feature = torch.zeros((CHUNK_SIZE, 2560), dtype=torch.float32) # unique for qwen3-4b
        return {
            "audio": [dummy_feature] * num_chunks,
        }


class SQwen3MultiModalProcessor(
        BaseMultiModalProcessor[SQwen3ProcessingInfo]):
    
    def get_hf_processor(self, **kwargs: object) -> SQwen3Processor:
        return self.ctx.init_processor(
            SQwen3Processor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object] = None,
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_embeds=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()

        audio_token = PLACEHOLDER_TOKEN
        audio_token_id = tokenizer.encode(audio_token)[0]

        def get_replacement_sqwen2(item_idx: int):
            return [audio_token_id] * CHUNK_SIZE
        
        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id] * CHUNK_SIZE,
                replacement=get_replacement_sqwen2,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    SQwen3MultiModalProcessor,
    info=SQwen3ProcessingInfo,
    dummy_inputs=SQwen3DummyInputsBuilder)
class SQwen3ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                         SupportsPP):

    vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return PLACEHOLDER_TOKEN * CHUNK_SIZE

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        audio_embeds = kwargs.get("audio_embeds", None)
        if audio_embeds is None:
            return []
        bsz, n_chunks, n_frames, n_dim = audio_embeds.shape
        audio_embeds = audio_embeds.view(bsz, n_chunks * n_frames, n_dim)
        return audio_embeds

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            audio_token_id = AUDIO_TOKEN_ID
            if isinstance(multimodal_embeddings, torch.Tensor):
                multimodal_embeddings = multimodal_embeddings.to(inputs_embeds)
            elif isinstance(multimodal_embeddings, list):
                multimodal_embeddings = [
                    x.to(inputs_embeds) if isinstance(x, torch.Tensor) else x
                    for x in multimodal_embeddings
                ]
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [audio_token_id] * CHUNK_SIZE
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.vllm_mapper)
