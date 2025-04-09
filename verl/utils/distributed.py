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
"""Utilities for distributed training."""
import os
import pickle


def initialize_global_process_group(timeout_second=36000):
    import torch.distributed
    from datetime import timedelta
    torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=timeout_second))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size

def broadcast_pyobj(data, rank, dist_group, src):
    import torch
    import torch.distributed as dist

    if rank == src:
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage)
        size_tensor = torch.LongTensor([tensor.numel()])
        dist.broadcast(size_tensor, src=src, group=dist_group)
        dist.broadcast(tensor, src=src, group=dist_group)
        return data
    else:
        size_tensor = torch.LongTensor([0])
        dist.broadcast(size_tensor, src=src, group=dist_group)
        tensor = torch.ByteTensor(size_tensor.item())
        dist.broadcast(tensor, src=src, group=dist_group)
        buffer = tensor.numpy().tobytes()
        return pickle.loads(buffer)
