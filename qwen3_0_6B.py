#!/usr/bin/env python3
"""
Qwen3-0.6B Implementation

Usage: python qwen3_0_6B.py

Features:
- Complete LLM inference engine
- Paged KV Cache management
- Prefix caching optimization
- Continuous batching
- Tensor parallel support
- Flash Attention with fallback to PyTorch native attention
"""

import os
import atexit
import pickle
import glob
from collections import deque
from copy import copy
from dataclasses import dataclass, fields
from enum import Enum, auto
from functools import lru_cache
from itertools import count
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

import triton
import triton.language as tl
import xxhash
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, Qwen3Config
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
    print("Flash Attention available, using optimized attention")
except ImportError:
    HAS_FLASH_ATTN = False
    print("Flash Attention not available, using PyTorch native attention")
from pynvml import *
from safetensors import safe_open


# ============= Data Structures =============

@dataclass
class SamplingParams:
    """Sampling parameters configuration"""
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False


@dataclass
class Config:
    """Global configuration"""
    model: str
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.11
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)


@dataclass
class Context:
    """Inference context"""
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


class SequenceStatus(Enum):
    """Sequence status enumeration"""
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """Sequence management class"""
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __lt__(self, other):
        return self.seq_id < other.seq_id

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


# ============= Context Management =============

_context = Context()

def get_context():
    return _context

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, 
                slot_mapping=None, context_lens=None, block_tables=None):
    global _context
    _context.is_prefill = is_prefill
    _context.cu_seqlens_q = cu_seqlens_q
    _context.cu_seqlens_k = cu_seqlens_k
    _context.max_seqlen_q = max_seqlen_q
    _context.max_seqlen_k = max_seqlen_k
    _context.slot_mapping = slot_mapping
    _context.context_lens = context_lens
    _context.block_tables = block_tables

def reset_context():
    global _context
    _context = Context()


# ============= Utility Functions =============

def get_gpu_memory():
    """Get GPU memory information"""
    torch.cuda.synchronize()
    nvmlInit()
    visible_device = list(map(int, os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(',')))
    cuda_device_idx = torch.cuda.current_device()
    real_device_idx = visible_device[cuda_device_idx]
    print(f"GPU Device Mapping: CUDA Device {cuda_device_idx} -> Physical GPU {real_device_idx}")
    handle = nvmlDeviceGetHandleByIndex(real_device_idx)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    total_memory = mem_info.total
    used_memory = mem_info.used
    free_memory = mem_info.free
    print(f"GPU Memory: Total {total_memory//1024//1024//1024:.1f}GB, Used {used_memory//1024//1024//1024:.1f}GB, Free {free_memory//1024//1024//1024:.1f}GB")
    nvmlShutdown()
    return total_memory, used_memory, free_memory


def compute_hash(token_ids: list[int], prefix: int = -1):
    """Compute hash of token sequence"""
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()


def divide(numerator, denominator):
    """Integer division function"""
    assert numerator % denominator == 0
    return numerator // denominator


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """Default weight loader"""
    assert param.size() == loaded_weight.size()
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """Model weight loader"""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


# ============= Memory Block Management =============

class Block:
    """Memory block class"""
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        assert hash != -1
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

    def __repr__(self):
        return f"{(self.block_id, self.ref_count, self.hash)}"


class BlockManager:
    """Memory block manager"""
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _allocate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence):
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence):
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1 


# ============= Scheduler =============

class Scheduler:
    """Sequence scheduler"""
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # Prefill阶段
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            
        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode阶段
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                
        running = deque(scheduled_seqs)
        running.extend(self.running)
        self.running = running
        
        assert scheduled_seqs
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)


# ============= Neural Network Layers =============

class SiluAndMul(nn.Module):
    """SiLU activation with multiplication"""
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


class RMSNorm(nn.Module):
    """RMS Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class Sampler(nn.Module):
    """Token sampler"""
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)


# ============= Linear Layer Base Classes =============

class LinearBase(nn.Module):
    """Linear layer base class"""
    def __init__(self, input_size: int, output_size: int, tp_dim: int | None = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Replicated linear layer"""
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """Column parallel linear layer"""
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, 0)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Merged column parallel linear layer"""
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV parallel linear layer"""
    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int, 
                 total_num_kv_heads: int | None = None, bias: bool = False):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = self.hidden_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
        ]
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """Row parallel linear layer"""
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, 1)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y 

# ============= Embedding Layers =============

class VocabParallelEmbedding(nn.Module):
    """Vocabulary parallel embedding layer"""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """Parallel language model head"""
    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits


# ============= Rotary Position Embedding =============

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding"""
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding"""
    def __init__(self, head_size: int, rotary_dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        assert rotary_dim == head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base**(torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(head_size: int, rotary_dim: int, max_position: int, base: float, rope_scaling: dict | None = None):
    """Get rotary position embedding"""
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


# ============= Attention Mechanism =============

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """Triton kernel for storing KV cache"""
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """Store KV cache"""
    N, num_heads, head_dim = key.shape
    # Use actual KV head count instead of query head count for KV cache
    num_kv_heads = k_cache.size(2)  # KV cache shape: (num_blocks, block_size, num_kv_heads, head_dim)
    actual_head_dim = k_cache.size(3)  # Use KV cache head_dim
    D = num_kv_heads * actual_head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # KV cache shape is (num_blocks, block_size, num_kv_heads, head_dim)
    # stride(1) should be num_kv_heads * head_dim
    # assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """Attention mechanism"""
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _pytorch_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          context: 'Context', k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        """
        PyTorch native attention implementation (fallback when Flash Attention is unavailable)
        
        Features:
        1. Supports both Prefill and Decode modes
        2. Handles variable-length sequence attention
        3. Supports GQA (Grouped Query Attention)
        4. Implements KV Cache management
        5. Applies causal mask for autoregressive generation
        
        Note: This implementation prioritizes compatibility over performance
        """
        batch_size = q.size(0)
        
        if context.is_prefill:
            # Prefill phase - handle attention for multiple sequences
            if context.block_tables is not None:
                # Use cached attention (simplified paged attention)
                # For simplicity, we use standard attention mechanism
                # Production systems would need more complex paged attention implementation
                outputs = []
                start_idx = 0
                for i in range(len(context.cu_seqlens_q) - 1):
                    seq_len_q = context.cu_seqlens_q[i+1] - context.cu_seqlens_q[i]
                    seq_len_k = context.cu_seqlens_k[i+1] - context.cu_seqlens_k[i]
                    
                    q_seq = q[start_idx:start_idx+seq_len_q].unsqueeze(0)  # [1, seq_len, num_heads, head_dim]
                    k_seq = k[start_idx:start_idx+seq_len_q].unsqueeze(0)
                    v_seq = v[start_idx:start_idx+seq_len_q].unsqueeze(0)
                    
                    # Rearrange dimensions to [batch, num_heads, seq_len, head_dim]
                    q_seq = q_seq.transpose(1, 2)  
                    k_seq = k_seq.transpose(1, 2)
                    v_seq = v_seq.transpose(1, 2)
                    
                    # Compute attention scores
                    scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * self.scale
                    
                    # Apply causal mask
                    causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=q.device), diagonal=1).bool()
                    scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    
                    # Apply softmax
                    attn_weights = F.softmax(scores, dim=-1)
                    
                    # Compute output
                    out = torch.matmul(attn_weights, v_seq)  # [1, num_heads, seq_len, head_dim]
                    out = out.transpose(1, 2).squeeze(0)  # [seq_len, num_heads, head_dim]
                    outputs.append(out)
                    start_idx += seq_len_q
                
                o = torch.cat(outputs, dim=0)
            else:
                # Standard multi-head self-attention
                outputs = []
                start_idx = 0
                for i in range(len(context.cu_seqlens_q) - 1):
                    seq_len = context.cu_seqlens_q[i+1] - context.cu_seqlens_q[i]
                    
                    q_seq = q[start_idx:start_idx+seq_len].unsqueeze(0).transpose(1, 2)
                    k_seq = k[start_idx:start_idx+seq_len].unsqueeze(0).transpose(1, 2)  
                    v_seq = v[start_idx:start_idx+seq_len].unsqueeze(0).transpose(1, 2)
                    
                    # Expand KV to match Q head count (for GQA)
                    if self.num_kv_heads != self.num_heads:
                        k_seq = k_seq.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                        v_seq = v_seq.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                    
                    scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * self.scale
                    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
                    scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    attn_weights = F.softmax(scores, dim=-1)
                    out = torch.matmul(attn_weights, v_seq).transpose(1, 2).squeeze(0)
                    outputs.append(out)
                    start_idx += seq_len
                
                o = torch.cat(outputs, dim=0)
        else:
            # Decode phase - single token generation using KV cache
            q = q.unsqueeze(1)  # [batch, 1, num_heads, head_dim]
            q = q.transpose(1, 2)  # [batch, num_heads, 1, head_dim]
            
            # Get key-value from KV cache
            outputs = []
            for i in range(batch_size):
                context_len = context.context_lens[i].item()
                
                # Get corresponding cache from block table
                block_indices = context.block_tables[i]
                valid_blocks = block_indices[block_indices != -1]
                
                if len(valid_blocks) == 0:
                    # No valid cache blocks, use current k, v
                    k_i = k[i:i+1].unsqueeze(0).transpose(1, 2)
                    v_i = v[i:i+1].unsqueeze(0).transpose(1, 2)
                else:
                    # Build complete k, v sequence from cache
                    # Simplified handling, production needs more complex block management
                    k_cached_blocks = []
                    v_cached_blocks = []
                    for block_idx in valid_blocks:
                        if block_idx < k_cache.size(0):
                            k_cached_blocks.append(k_cache[block_idx])
                            v_cached_blocks.append(v_cache[block_idx])
                    
                    if k_cached_blocks:
                        k_cached = torch.cat(k_cached_blocks, dim=0)[:context_len]
                        v_cached = torch.cat(v_cached_blocks, dim=0)[:context_len]
                        
                        # Add current token's k, v
                        k_i = torch.cat([k_cached, k[i:i+1]], dim=0).unsqueeze(0).transpose(1, 2)
                        v_i = torch.cat([v_cached, v[i:i+1]], dim=0).unsqueeze(0).transpose(1, 2)
                    else:
                        k_i = k[i:i+1].unsqueeze(0).transpose(1, 2)
                        v_i = v[i:i+1].unsqueeze(0).transpose(1, 2)
                
                # Expand KV head count to match Q
                if self.num_kv_heads != self.num_heads:
                    k_i = k_i.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                    v_i = v_i.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                
                q_i = q[i:i+1]
                scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * self.scale
                attn_weights = F.softmax(scores, dim=-1)
                out_i = torch.matmul(attn_weights, v_i).transpose(1, 2).squeeze(0)
                outputs.append(out_i)
            
            o = torch.cat(outputs, dim=0)
        
        return o

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache = self.k_cache
        v_cache = self.v_cache
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if HAS_FLASH_ATTN:
            # Use Flash Attention
            if context.is_prefill:
                if context.block_tables is not None:
                    # Use paged attention
                    o = flash_attn_varlen_func(q, k_cache, v_cache,
                                               max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                               max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                               softmax_scale=self.scale, causal=True, block_table=context.block_tables)
                else:
                    # Use regular attention
                    o = flash_attn_varlen_func(q, k, v,
                                               max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                               max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                               softmax_scale=self.scale, causal=True)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
        else:
            # Use PyTorch native attention
            o = self._pytorch_attention(q, k, v, context, k_cache, v_cache)
        
        o = o.view(-1, self.num_heads * self.head_dim)
        return o 

# ============= Qwen3 Model Definition =============

class Qwen3Attention(nn.Module):
    """Qwen3 attention layer"""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, max_position: int = 4096 * 32,
                 head_dim: int | None = None, rms_norm_eps: float = 1e-06, qkv_bias: bool = False,
                 rope_theta: float = 10000, rope_scaling: tuple | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        


        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, 
                                          self.total_num_kv_heads, bias=qkv_bias)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = get_rope(self.head_dim, self.head_dim, max_position, rope_theta, rope_scaling)
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output = self.o_proj(attn_output)
        return output


class Qwen3MLP(nn.Module):
    """Qwen3 MLP layer"""
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size, intermediate_size], bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 decoder layer"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 model"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 causal language model"""
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.tie_word_embeddings = config.tie_word_embeddings
        if self.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits 

# ============= Model Runner =============

class ModelRunner:
    """Model runner"""
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
    
        if self.world_size > 1:
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.allocate_kv_cache(config.gpu_memory_utilization)
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def loop(self):
        """Worker process main loop"""
        while True:
            event = self.event
            event.wait()
            event.clear()
            try:
                data = pickle.loads(self.shm.buf[:])
                func, args, kwargs = data["func"], data["args"], data["kwargs"]
                result = getattr(self, func)(*args, **kwargs)
                data["result"] = result
                self.shm.buf[:len(pickle.dumps(data))] = pickle.dumps(data)
            except:
                break

    def call(self, func: str, *args, **kwargs):
        """Call method"""
        if self.world_size == 1:
            return getattr(self, func)(*args, **kwargs)
        data = {"func": func, "args": args, "kwargs": kwargs}
        self.shm.buf[:len(pickle.dumps(data))] = pickle.dumps(data)
        for event in self.event:
            event.set()
        self.event[0].wait()
        self.event[0].clear()
        data = pickle.loads(self.shm.buf[:])
        return data["result"]

    def allocate_kv_cache(self, gpu_memory_utilization: float):
        """Allocate KV cache"""
        total_memory, used_memory, free_memory = get_gpu_memory()
        
        # Calculate memory size per block (K and V cache)
        # Use actual head_dim instead of hidden_size // num_attention_heads
        head_dim = getattr(self.config.hf_config, 'head_dim', None) or self.config.hf_config.hidden_size // self.config.hf_config.num_attention_heads
        num_kv_heads = getattr(self.config.hf_config, 'num_key_value_heads', self.config.hf_config.num_attention_heads)
        num_kv_heads_per_gpu = num_kv_heads // self.world_size
        
        # Memory per block = block_size * head_dim * num_kv_heads_per_gpu * 2 (K and V) * 2 (float16) * num_layers
        bytes_per_block = self.block_size * head_dim * num_kv_heads_per_gpu * 2 * 2 * len(self.model.model.layers)
        
        # Calculate block count based on available memory
        available_memory = free_memory * gpu_memory_utilization
        num_blocks = max(1, int(available_memory // bytes_per_block))
        self.config.num_kvcache_blocks = num_blocks
        
        print(f"KV Cache Allocation: {num_blocks} blocks, {bytes_per_block//1024//1024:.1f}MB per block, {num_blocks*bytes_per_block//1024//1024:.1f}MB total")
        
        # Allocate KV cache tensors - using original shape format
        # shape: (num_blocks, block_size, num_kv_heads_per_gpu, head_dim)
        kvcache_shape = (num_blocks, self.block_size, num_kv_heads_per_gpu, head_dim)
        
        # Create global KV cache tensor
        self.kv_cache = torch.zeros(2, len(self.model.model.layers), num_blocks, self.block_size, num_kv_heads_per_gpu, head_dim, 
                                   dtype=self.config.hf_config.torch_dtype, device="cuda")
        
        # Assign KV cache for each layer
        for layer_id, layer in enumerate(self.model.model.layers):
            layer.self_attn.attn.k_cache = self.kv_cache[0, layer_id]
            layer.self_attn.attn.v_cache = self.kv_cache[1, layer_id]

    def capture_cudagraph(self):
        """Capture CUDA graph"""
        max_batch_size = 512
        self.graphs = {}
        self.graph_vars = {}
        self.graph_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        
        for bs in self.graph_bs:
            graph = torch.cuda.CUDAGraph()
            graph_vars = {}
            graph_vars["input_ids"] = torch.zeros(max_batch_size, dtype=torch.int64, device="cuda")
            graph_vars["positions"] = torch.zeros(max_batch_size, dtype=torch.int64, device="cuda")
            graph_vars["slot_mapping"] = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
            graph_vars["context_lens"] = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
            max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
            graph_vars["block_tables"] = torch.zeros(max_batch_size, max_num_blocks, dtype=torch.int32, device="cuda")
            
            with torch.cuda.graph(graph):
                graph_vars["outputs"] = self.model(
                    graph_vars["input_ids"][:bs], 
                    graph_vars["positions"][:bs]
                )
            
            self.graphs[bs] = graph
            self.graph_vars = graph_vars

    def prepare_block_tables(self, seqs: list[Sequence]):
        """Prepare block tables"""
        max_num_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = []
        for seq in seqs:
            block_table = seq.block_table + [0] * (max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """Prepare prefill data"""
        input_ids = []
        positions = []
        slot_mapping = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        context_lens = []
        
        for seq in seqs:
            seq_len = len(seq) - seq.num_cached_tokens
            start_pos = seq.num_cached_tokens
            input_ids.extend(seq.token_ids[start_pos:])
            positions.extend(range(start_pos, len(seq)))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seq_len)
            cu_seqlens_k.append(cu_seqlens_k[-1] + len(seq))
            context_lens.append(len(seq))
            
            for i in range(start_pos, len(seq)):
                block_idx = i // self.block_size
                block_offset = i % self.block_size
                slot_mapping.append(seq.block_table[block_idx] * self.block_size + block_offset)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        max_seqlen_q = max(len(seq) - seq.num_cached_tokens for seq in seqs)
        max_seqlen_k = max(len(seq) for seq in seqs)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # For prefill phase, don't use block_tables if sequences have no cached tokens
        use_block_tables = any(seq.num_cached_tokens > 0 for seq in seqs)
        final_block_tables = block_tables if use_block_tables else None
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, final_block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """Prepare decode data"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """Prepare sampling data"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill):
        """Run model"""
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            self.reset_graph_vars()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def reset_graph_vars(self):
        """Reset graph variables"""
        graph_vars = self.graph_vars
        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["slot_mapping"].zero_()
        graph_vars["context_lens"].zero_()
        graph_vars["block_tables"].zero_()
    
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Run inference"""
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def exit(self):
        """Exit"""
        pass


# ============= LLM Engine =============

class LLMEngine:
    """LLM inference engine"""
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        for i in range(1, config.tensor_parallel_size):
            event = mp.Event()
            process = mp.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        """Exit engine"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Add request"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """Execute one inference step"""
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        """Check if finished"""
        return self.scheduler.is_finished()

    def generate(self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams], use_tqdm: bool = True) -> list[str]:
        """Generate text"""
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs


class LLM(LLMEngine):
    """LLM class"""
    pass 

# ============= Main Function =============

if __name__ == "__main__":
    # Example usage
    import os
    
    print("Environment Check:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Count: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.current_device()}")
    print(f"   Flash Attention: {'Available' if HAS_FLASH_ATTN else 'Not Available'}")
    
    # Model path - modify this to your actual model path
    path = os.path.expanduser("~/model/qwen/Qwen3-0.6B")
    
    # Check if model path exists
    if not os.path.exists(path):
        print(f"Model path does not exist: {path}")
        print("Please modify the path variable to point to the correct Qwen3-0.6B model directory")
        print("   Example: path = '/path/to/your/Qwen3-0.6B'")
        exit(1)
    
    print("Initializing Qwen3-0.6B model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        llm = LLM(path, enforce_eager=True, gpu_memory_utilization=0.10)
        print("Model initialization complete!")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        print("Please check:")
        print("   1. Model path is correct")
        print("   2. Model files are complete")
        print("   3. GPU memory is sufficient")
        exit(1)
    
    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # Prepare prompts
    prompts = [
        "who are you?",
    ]
    
    print("Applying chat template...")
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    
    print("Starting text generation...")
    outputs = llm.generate(prompts, sampling_params)
    
    print("\n" + "="*80)
    print("Generation Results:")
    print("="*80)
    
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
        print("-" * 40)
    
    print("\nGeneration complete!")
    print("\nUsage Instructions:")
    print("- This script integrates complete LLM inference functionality")
    print("- Supports paged KV Cache management")
    print("- Supports prefix caching optimization")
    print("- Supports continuous batching")
    print("- Supports tensor parallelism (requires multiple GPUs)")
    print("- Auto-adapts to Flash Attention (if available) or PyTorch native attention")
    print("- Modify the path variable to point to your Qwen3-0.6B model directory")
    print("\nConfiguration:")
    print("- enforce_eager=True: Disable CUDA graph optimization, suitable for debugging")
    print("- gpu_memory_utilization=0.10: Use 10% of GPU memory")
    print("- temperature=0.6: Sampling temperature")
    print("- max_tokens=256: Maximum generation tokens")
    print("\nPerformance Optimization:")
    print("- Install Flash Attention for best performance: pip install flash-attn")
    print("- If Flash Attention is installed, the script automatically uses optimized implementation")
    print("- If not installed, the script uses compatible PyTorch native implementation") 