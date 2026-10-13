# AOT ID: ['1_inference']
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


# kernel path: /tmp/tmpaldds34w/pl/cplzgm7bimlhtnw2xasse33itakx6qblhaqk7oislqu3fhit757r.py
# Topologically Sorted Source Nodes: [linear, relu], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   linear => add_tensor_1
#   relu => relu
# Graph fragment:
#   %arg1_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm_default_1 : Tensor "f32[4, 32][32, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %add_tensor_1 : Tensor "f32[4, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %mm_default_1), kwargs = {})
#   %relu : Tensor "f32[4, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
#   return %relu
triton_poi_fused_addmm_relu_0 = async_compile.triton('triton_poi_fused_addmm_relu_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 1664}, 'backend_hash': '25ED88F5935B940EC1927250A0CAECF41F6A0D2252E814102945959BFC20A347', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2, tl.PropagateNan.ALL)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
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
        assert_size_stride_grouped((arg2_1, arg0_1), ((4, 16), (32, 16)), ((16, 1), (16, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf0 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.t, aten.addmm]
            extern_kernels.mm(arg2_1, reinterpret_tensor(arg0_1, (16, 32), (1, 16), 0), out=buf0)
            del arg0_1
            del arg2_1
            assert_size_stride(arg1_1, (32, ), (1, ), 'input')
            buf1 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [linear, relu], Original ATen: [aten.addmm, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_0.run(buf1, arg1_1, 128, stream=raw_stream0)
            del arg1_1
            assert_size_stride(arg3_1, (32, 32), (32, 1), 'input')
            buf2 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, relu, linear_1], Original ATen: [aten.addmm, aten.relu, aten.t]
            extern_kernels.mm(buf1, reinterpret_tensor(arg3_1, (32, 32), (1, 32), 0), out=buf2)
            del arg3_1
            del buf1
            assert_size_stride(arg4_1, (32, ), (1, ), 'input')
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [linear_1, relu_1], Original ATen: [aten.addmm, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_0.run(buf3, arg4_1, 128, stream=raw_stream0)
            del arg4_1
            assert_size_stride_grouped((arg6_1, arg5_1), ((8, ), (8, 32)), ((1, ), (32, 1)), 'input')
            buf4 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, relu_1, linear_2], Original ATen: [aten.addmm, aten.relu, aten.t]
            extern_kernels.addmm(arg6_1, buf3, reinterpret_tensor(arg5_1, (32, 8), (1, 32), 0), alpha=1, beta=1, out=buf4)
            del arg5_1
            del arg6_1
            del buf3
        return (buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((32, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
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
