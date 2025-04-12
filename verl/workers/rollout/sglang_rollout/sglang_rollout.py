# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
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
from collections import OrderedDict
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, List
from uuid import uuid4
import numpy as np
from omegaconf import DictConfig
from tensordict import TensorDict
from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length, pad_2d_list_to_length
# from sglang.srt.entrypoints.verl_engine import VerlEngine
from verl.third_party.sglang.entrypoint import VerlEngine
from torch.distributed.device_mesh import init_device_mesh
from sglang.srt.sampling.sampling_params import SamplingParams
from verl.utils.distributed import broadcast_pyobj
from verl.third_party.sglang import parallel_state as sglang_ps
import torch.distributed
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from torch import nn


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


# NOTE(linjunrong): adhoc
def _post_process_outputs(tokenizer, output):

    def _map_each_response(l):
        # output_token_ids = torch.tensor(l['token_ids'])
        log_probs = []
        output_token_ids = []
        for log_prob, token_ids, _ in l["meta_info"]["output_token_logprobs"]:
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

def _post_process_partial_outputs(input_ids, output):
    for l in output["meta_info"]["output_token_logprobs"]:
        input_ids.append(l[1])
    return input_ids


class SGLangRollout(BaseRollout):

    def __init__(
        self,
        actor_module: nn.Module | str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        **kwargs,
    ):
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

        assert not (not config.enforce_eager and
                    config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert (tensor_parallel_size <= torch.distributed.get_world_size()
               ), "tensor parallel size should be less than or equal to the world size"

        if kwargs.get("train_tp", None) is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        assert (model_hf_config.max_position_embeddings >= config.prompt_length +
                config.response_length), "model context length should be greater than total sequence length"

        tp_size = tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        device_mesh_kwargs = dict(
            mesh_shape=(world_size // tp_size, tp_size, 1),
            mesh_dim_names=["dp", "tp", "pp"],
        )
        device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)
        # device_mesh_device = init_device_mesh("cuda", **device_mesh_kwargs)

        # get tp_rank of this process in this tp group
        visible_devices = [None] * device_mesh_cpu.size(1)
        torch.distributed.all_gather_object(visible_devices, os.environ["CUDA_VISIBLE_DEVICES"],
                                            device_mesh_cpu.get_group("tp"))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)

        self.inference_engine = VerlEngine(
            model_path=actor_module,
            dtype=config.dtype,
            mem_fraction_static=config.gpu_memory_utilization,
            device_mesh_cpu=device_mesh_cpu["tp"],
            enable_memory_saver=not ("AMD" in torch.cuda.get_device_name()),
            base_gpu_id=0,
            gpu_id_step=1,
            # NOTE(Chenyang): if you want to debug the sglang engine
            # please set the following parameters
            # Otherwise, it will make the engine run too slow
            # log_level="INFO",
            # log_requests=True,
            # log_requests_level=2,
            max_running_requests=128,
            cuda_graph_max_bs=128,
            enable_mixed_chunk=True,
            stream_interval=256,
            enable_torch_compile=False,
        )

        # offload
        self.inference_engine.release_memory_occupation()

        kwargs = dict(n=1,
                      max_new_tokens=config.response_length,
                      presence_penalty=0.0,
                      frequency_penalty=0.0,
                      repetition_penalty=1.0)
        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"kwargs: {kwargs}")
        self.sampling_params = kwargs

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.partial_rollout = config.get("partial_rollout", True)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if key in self.sampling_params:
                    old_value = self.sampling_params[key]
                    old_sampling_params_args[key] = old_value
                    self.sampling_params[key] = value
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            self.sampling_params[key] = value

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # if self.config.free_cache_engine:

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get("do_sample", True)
        if not do_sample:
            # kwargs = {
            #     'top_p': 1.0,
            #     'top_k': -1,
            #     'min_p': 0.0,
            #     'temperature': 0,
            #     'n': 1  # if greedy, only 1 response
            # }
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=1,
                max_new_tokens=self.config.response_length,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        is_validate = prompts.meta_info.get('validate', False)
        if is_validate:
            kwargs.update(self.config.val_kwargs)
        batch_sampling_params = prompts.meta_info.get('sampling_params', {})
        kwargs.update(batch_sampling_params)
        n = kwargs.get('n', self.config.n)
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            print(f"{self.sampling_params=}")
            output = self.inference_engine.generate(
                prompt=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                return_logprob=True,
                input_ids=idx_list,
            )

        out = _post_process_outputs(self.tokenizer, output)

        response = out[0].to(idx.device)
        log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)
        if n > 1 and do_sample:
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
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free cache engine
        if self.config.free_cache_engine and self.inference_engine._engine is not None:
            self.inference_engine._engine.tokenizer_manager.flush_cache()

        return DataProto(batch=batch)

    @torch.no_grad()
    def feed_group_cache(self, prompts: DataProto, **kwargs):
        self.group_iter = 0
        self.group_cache = {}
        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        gids = prompts.batch['gids']
        bsz = prompts.batch['input_ids'].size(0)
        n = kwargs.get('n', self.config.n)

        # Generate rids
        all_rids = {}
        for i in range(bsz):
            oid = f"req_{i}_{gids[i]}"
            all_rids[oid] = []
            for j in range(n):
                nid = f"{oid}_nid{uuid4().hex[:8]}"
                all_rids[oid].append(nid)

        # Sync the rids across all the tp ranks
        all_rids = broadcast_pyobj(
            data=all_rids,
            rank=torch.distributed.get_rank(),
            dist_group=self.inference_engine._device_mesh_cpu.get_group(),
            src=self.inference_engine._device_mesh_cpu.mesh[0].item(),
        )

        for i in range(bsz):
            idx = prompts.batch['input_ids'][i]
            attention_mask = prompts.batch['attention_mask'][i]
            position_ids = prompts.batch['position_ids'][i]
            idx_ = _pre_process_inputs(self.pad_token_id, idx)
            gid = gids[i]
            oid = f"req_{i}_{gid}"
            self.group_cache[oid] = {}
            for j in range(n):
                rid = all_rids[oid][j]
                self.group_cache[oid][rid] = {
                    'idx': idx,
                    'processed_idx': idx_,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids,
                    'gid': gid,
                    'rid': rid,
                    'output': None,
                }
        self.group_meta = prompts.meta_info
        if prompts.meta_info.get('group_shuffle', False):
            n_groups = prompts.meta_info['n_groups']
            self.mini_bsz = bsz // n_groups
        elif prompts.meta_info.get('oversubscribe', False):
            n_over = prompts.meta_info['n_over']
            self.mini_bsz = bsz // n_over

    @torch.no_grad()
    def get_cached_outputs(self):
        cache_dict = {}
        for oid, cache_dict in self.group_cache.items():
            for rid, cached_state in cache_dict.items():
                if cached_state.get('cached_ids', None) is not None:
                    cache_dict[rid] = cached_state['cached_ids']
        return cache_dict
        
    @torch.no_grad()
    def prepare_batch(self):
        idx_list = []
        rids = []
        flat_group_cache = [v for cache_dict in self.group_cache.values() for v in cache_dict.values()]
        rid_map = {}
        for v in flat_group_cache:
            if v['output'] is None or not self.partial_rollout:
                rid = v['rid']
                oid = rid.split('_nid')[0]
                rid_ = f'{oid}_nid{uuid4().hex[:8]}'
                rid_map[rid] = rid_
                # Prepare inputs for the engine
                processed_idx = v['processed_idx']
                if self.partial_rollout:
                    processed_idx.extend(v.get('cached_ids', []))
                idx_list.append(processed_idx)
                rids.append(rid_)
        oids = [rid.split('_nid')[0] for rid in rids]
        num_oids = len(set(oids))
        # Sync the rids across all the tp ranks
        rid_map = broadcast_pyobj(
            data=rid_map,
            rank=torch.distributed.get_rank(),
            dist_group=self.inference_engine._device_mesh_cpu.get_group(),
            src=self.inference_engine._device_mesh_cpu.mesh[0].item(),
        )
        for old_rid, new_rid in rid_map.items():
            oid = old_rid.split('_nid')[0]
            v = self.group_cache[oid].pop(old_rid)
            v['rid'] = new_rid
            self.group_cache[oid][new_rid] = v

        return idx_list, rids, num_oids

    def convert_output_id_to_logprob(self, output_ids):
        log_probs = []
        for output_id in output_ids:
            log_probs.append((0., output_id, None))
        return log_probs
    

    @torch.no_grad
    def collate_responses(self, output, batch_size):
        cache = self.group_cache
        for oid, out_dict in output.items():
            for rid, output in out_dict.items():
                finish_reason = output['meta_info']['finish_reason']
                finished = finish_reason['type'] != 'abort'
                if finished:
                    if self.partial_rollout:
                        cached_log_probs = self.convert_output_id_to_logprob(self.group_cache[oid][rid].get('cached_ids', []))
                        output['output_token_logprobs'] = output.get('output_token_logprobs', []) + cached_log_probs
                    cache[oid][rid]['output'] = output
                elif self.partial_rollout:
                    cache[oid][rid]['cached_ids'] = output.get('output_ids', [])
        ret = []
        idx = []
        attention_mask = []
        position_ids = []
        gids = []
        rids = []
        flat_group_cache = [v for cache_dict in cache.values() for v in cache_dict.values()]
        n_finished = {}
        finished_cache = {}
        for v in flat_group_cache:
            rid = v['rid']
            oid = rid.split('_nid')[0]
            if v['output'] is not None:
                n_finished[oid] = n_finished.get(oid, 0) + 1
                if n_finished[oid] == len(cache[oid]):
                    finished_cache[oid] = cache.pop(oid)
            if len(finished_cache.keys()) >= batch_size:
                break
        for oid, cache_dict in finished_cache.items():
            ret.extend([v['output'] for v in cache_dict.values()])
            idx.extend([v['idx'] for v in cache_dict.values()])
            attention_mask.extend([v['attention_mask'] for v in cache_dict.values()])
            position_ids.extend([v['position_ids'] for v in cache_dict.values()])
            gids.extend([v['gid'] for v in cache_dict.values()])
            rids.extend([v['rid'] for v in cache_dict.values()])
        self.group_cache = cache
        return ret, idx, attention_mask, position_ids, gids, rids
    
    @torch.no_grad()
    def generate_sequences_ingroup(self, **kwargs) -> DataProto:
        # if self.config.free_cache_engine:
        batch_size = min(self.mini_bsz, len(self.group_cache))
        idx_list = []
        attention_mask = []
        position_ids = []
        gids = []
        idx_list, rids, num_oids = self.prepare_batch()
        print(f"num_oids: {num_oids}, batch_size: {batch_size}, fed ids: {len(idx_list)}")
        num_returns = min(num_oids, batch_size)
        do_sample = self.group_meta.get('do_sample', True)
        eos_token_id = self.group_meta['eos_token_id']

        if not do_sample:
            # kwargs = {
            #     'top_p': 1.0,
            #     'top_k': -1,
            #     'min_p': 0.0,
            #     'temperature': 0,
            #     'n': 1  # if greedy, only 1 response
            # }
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=1,
                max_new_tokens=self.config.response_length,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            print(f"{self.sampling_params=}")
            outputs = self.inference_engine.generate(
                prompt=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                return_logprob=True,
                input_ids=idx_list,
                rid=rids,
                num_returns=num_returns,
            )
            output, idx, attention_mask, position_ids, gids = self.collate_responses(outputs, batch_size)
            device_ = idx[0].device
            idx = torch.stack(idx, dim=0).to(device_)
            attention_mask = torch.stack(attention_mask, dim=0)
            position_ids = torch.stack(position_ids, dim=0)
            gids = torch.stack(gids, dim=0).cpu()
                
        out = _post_process_outputs(self.tokenizer, output)

        response = out[0].to(idx.device)
        log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)
        if self.config.n > 1 and do_sample:
        #     idx = idx.repeat_interleave(self.config.n, dim=0)
        #     attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
        #     position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
        #     gids = gids.repeat_interleave(self.config.n, dim=0)
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
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "gids": gids,
            },
            batch_size=batch_size,
        )
        non_tensor_batch = {
            "rids": rids,
        }
        meta_info = {
            "cached_outputs": self.get_cached_outputs(),
        }

        # free cache engine
        if self.config.free_cache_engine and self.inference_engine._engine is not None:
            self.inference_engine._engine.tokenizer_manager.flush_cache()

        self.group_iter += 1

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
