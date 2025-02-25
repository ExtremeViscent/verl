# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, List
from omegaconf import DictConfig
from tensordict import TensorDict
from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.third_party.sglang.entrypoint import EngineFragment
from sglang.srt.distributed import ParallelProcessGroups
from torch.distributed.device_mesh import init_device_mesh
from sglang.srt.sampling.sampling_params import SamplingParams
from verl.third_party.sglang import parallel_state as sglang_ps
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from uuid import uuid4

if TYPE_CHECKING:
    from torch import nn


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


# NOTE(ljr): adhoc
def _post_process_outputs(tokenizer, output):

    def _map_each_response(l):
        # output_token_ids = torch.tensor(l['token_ids'])
        log_probs = []
        output_token_ids = []
        for log_prob, token_ids, _ in l['meta_info']['output_token_logprobs']:
            log_probs.append(log_prob)
            output_token_ids.append(token_ids)
        log_probs = torch.tensor(log_probs)
        output_token_ids = torch.tensor(output_token_ids)
        return output_token_ids, log_probs

    out_map = map(lambda x: _map_each_response(x), output)
    batched_output_token_ids = []
    batched_logprobs = []
    for output_token_ids, log_probs in out_map:
        batched_output_token_ids.append(output_token_ids)
        batched_logprobs.append(log_probs)
    pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    batched_output_token_ids = pad_sequence(batched_output_token_ids, batch_first=True, padding_value=pad_token_id)
    if len(batched_logprobs) > 0:
        batched_logprobs = pad_sequence(batched_logprobs, batch_first=True, padding_value=pad_token_id)
    return batched_output_token_ids, batched_logprobs


class SGLangRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module | str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A SGLang rollout. It requires the module is supported by the SGLang.

        Args:
            actor_module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in SGLang
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            sglang_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        tp_size = tensor_parallel_size
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        device_mesh_kwargs = dict(mesh_shape=(world_size // tp_size, tp_size, 1), mesh_dim_names=["dp", "tp", "pp"])
        device_mesh_device = init_device_mesh("cuda", **device_mesh_kwargs)
        device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        # get tp_rank of this process in this tp group
        tp_rank = device_mesh_device.get_local_rank("tp")

        self.inference_engine = EngineFragment(
            model_path=actor_module,
            tp_size=tensor_parallel_size,
            tp_rank=tp_rank,
            gpu_id=local_rank,
            nccl_port=23456,
            enable_mixed_chunk=True,
            skip_tokenizer_init=False,
            mem_fraction_static=config.gpu_memory_utilization,
            dtype=config.dtype,
            max_running_requests=128,
            cuda_graph_max_bs=128,
            parallel_process_groups=ParallelProcessGroups.from_devices_meshes(
                device_mesh_device=device_mesh_device,
                device_mesh_cpu=device_mesh_cpu,
                dim_tp="tp",
                dim_pp="pp",
            ),
        )
        sglang_ps.ensure_model_parallel_initialized(tp_size, 1)

        #offload
        self.inference_engine.release_gpu_occupation()

        kwargs = dict(
            n=1,
            max_new_tokens=config.response_length,
        )
        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"kwargs: {kwargs}")
        self.sampling_params = kwargs

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        self.group_cache = {}
        self.group_kwargs = {}
        self.group_meta = {}

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def feed_group_cache(self, prompts: DataProto, **kwargs):
        
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        gids = prompts.batch['gids']
        
        for i in range(prompts.batch['input_ids'].size(0)):
            idx = prompts.batch['input_ids'][i]
            attention_mask = prompts.batch['attention_mask'][i]
            position_ids = prompts.batch['position_ids'][i]
            idx_ = _pre_process_inputs(self.pad_token_id, idx)
            rid = uuid4()
            gid = gids[i]
            self.group_cache[rid] = {
                'idx': idx,
                'processed_idx': idx_,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'gid': gid
            }
        self.group_meta = prompts.meta_info

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            self.group_kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

    def generate_sequences_ingroup(self, mini_bsz: DataProto) -> DataProto:
        mini_bsz = mini_bsz.meta_info.get('mini_bsz', 32)
        idx_list = []
        rids = []
        idx = []
        attention_mask = []
        position_ids = []
        gids = []
        for rid, v in self.group_cache.items():
            idx_list.append(v['processed_idx'])
            rids.append(rid)
        do_sample = self.group_meta.get('do_sample', True)
        eos_token_id = self.group_meta['eos_token_id']

        with self.update_sampling_params(**self.group_kwargs):
            output, completed_rids, remain_rids = self.inference_engine.generate(
                prompt=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                return_logprob=True,
                input_ids=idx_list,
                rid=rids,
                num_return_sequences=mini_bsz
            )
            print(f"completed_rids: {len(completed_rids)}, remain_rids: {len(remain_rids)}")

        for rid in completed_rids:
            idx.append(self.group_cache[rid]['idx'])
            attention_mask.append(self.group_cache[rid]['attention_mask'])
            position_ids.append(self.group_cache[rid]['position_ids'])
            gids.append(self.group_cache[rid]['gid'])
        device_ = idx[0].device
        idx = torch.stack(idx, dim=0).to(device_)
        
        self.group_cache = {rid: self.group_cache[rid] for rid in remain_rids}
            
        out = _post_process_outputs(self.tokenizer, output)
        # print(out)
        response = out[0].to(idx.device)
        log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'gids': gids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine._entrypoint._scheduler.flush_cache()

        return DataProto(batch=batch)


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # if self.config.free_cache_engine:

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)
        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompt=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                return_logprob=True,
                input_ids=idx_list)

        out = _post_process_outputs(self.tokenizer, output)
        # print(out)
        response = out[0].to(idx.device)
        log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine._entrypoint._scheduler.flush_cache()

        return DataProto(batch=batch)
