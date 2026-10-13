"""Triton kernel + the @triton_op registration that makes it visible to torch.export.

  triton_exp::fused_mlp2  - FOUR PyTorch ops (linear, relu, linear, relu) inside ONE kernel
                            (a stand-in for something like a fused attention block)

What @triton_op does: registers the function as a custom op under torch.ops.triton_exp.*,
so tracing sees a single opaque node instead of trying to trace the kernel body.
wrap_triton(...) lets Dynamo/Inductor still understand the launch (grid, args) so that
Inductor can emit the user kernel as-is in the generated code.
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


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


@triton_op("triton_exp::fused_mlp2", mutates_args={})
def fused_mlp2(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
               w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    H1, H2 = w1.shape[0], w2.shape[0]
    out = torch.empty(M, H2, device=x.device, dtype=x.dtype)
    BLOCK_M = 16  # tl.dot needs >= 16
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    wrap_triton(_mlp2_kernel)[grid](x, w1, b1, w2, b2, out, M, K=K, H1=H1, H2=H2, BLOCK_M=BLOCK_M)
    return out
