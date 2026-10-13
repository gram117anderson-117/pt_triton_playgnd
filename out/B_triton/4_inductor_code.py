# AOT ID: ['3_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# Original path: /home/grammy/agent_playground/triton_exp/kernels.py:17
_mlp2_kernel_0 = async_compile.triton('_mlp2_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'declared_constexpr_names': ['K', 'H1', 'H2', 'BLOCK_M'], 'kernel_name': '_mlp2_kernel_0', 'backend_hash': '25ED88F5935B940EC1927250A0CAECF41F6A0D2252E814102945959BFC20A347', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'x_ptr': '*fp32', 'w1_ptr': '*fp32', 'b1_ptr': '*fp32', 'w2_ptr': '*fp32', 'b2_ptr': '*fp32', 'out_ptr': '*fp32', 'M': 'i32', 'K': 'constexpr', 'H1': 'constexpr', 'H2': 'constexpr', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'K': 16, 'H1': 32, 'H2': 32, 'BLOCK_M': 16}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _mlp2_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr, M,
                 K: tl.constexpr, H1: tl.constexpr, H2: tl.constexpr, BLOCK_M: tl.constexpr):
    # Each program handles BLOCK_M rows and keeps the whole intermediate activation on-chip.
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rk, r1, r2 = tl.arange(0, K), tl.arange(0, H1), tl.arange(0, H2)
    x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M, other=0.0)
    # nn.Linear stores weight as [out, in]; load it transposed as [in, out] for tl.dot.
    w1 = tl.load(w1_ptr + r1[None, :] * K + rk[:, None])
    h = tl.dot(x, w1, input_precision="ieee") + tl.load(b1_ptr + r1)[None, :]   # linear 1
    h = tl.maximum(h, 0.0)                                                       # relu 1
    w2 = tl.load(w2_ptr + r2[None, :] * H1 + r1[:, None])
    o = tl.dot(h, w2, input_precision="ieee") + tl.load(b2_ptr + r2)[None, :]   # linear 2
    o = tl.maximum(o, 0.0)                                                       # relu 2
    tl.store(out_ptr + rm[:, None] * H2 + r2[None, :], o, mask=rm[:, None] < M)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
            assert_size_stride_grouped((arg4_1, arg0_1, arg1_1, arg2_1, arg3_1), ((4, 16), (32, 16), (32, ), (32, 32), (32, )), ((16, 1), (16, 1), (1, ), (32, 1), (1, )), 'input')
            arg4_1 = copy_if_misaligned(arg4_1)
            # Topologically Sorted Source Nodes: [h], Original ATen: []
            raw_stream0 = get_raw_stream(0)
            _mlp2_kernel_0.run(arg4_1, arg0_1, arg1_1, arg2_1, arg3_1, buf0, 4, 1, 1, 1, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
            del arg4_1
            assert_size_stride_grouped((arg6_1, arg5_1), ((8, ), (8, 32)), ((1, ), (32, 1)), 'input')
            buf2 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.t, aten.addmm]
            extern_kernels.addmm(arg6_1, buf0, reinterpret_tensor(arg5_1, (32, 8), (1, 32), 0), alpha=1, beta=1, out=buf2)
            del arg5_1
            del arg6_1
            del buf0
        return (buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((32, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((4, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((8, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
