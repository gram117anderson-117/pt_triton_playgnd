# Comparison

- **A_pytorch** - Pure PyTorch 3-layer MLP: fc1 -> relu -> fc2 -> relu -> fc3, one output. No Triton. The reference.
- **B_triton** - Triton kernel for fc1..relu2: Same MLP and weights, one output. `@triton_op triton_exp::fused_mlp2` does linear+relu+linear+relu in ONE kernel (4 PyTorch ops hidden in one node), then `fc3` is plain PyTorch. Compare with A_pytorch to see the Triton node replace those four ops.

Outputs match across experiments (checked in `main`). Cells list compute ops in graph order.


## 1 fx symbolic_trace

| experiment | #ops | param inputs | ops |
|---|---|---|---|
| A_pytorch | 5 | 0 | `Linear` -> `ReLU` -> `Linear` -> `ReLU` -> `Linear` |
| B_triton | 2 | 4 | `triton_exp.fused_mlp2` -> `Linear` |

Op counts: **A_pytorch**: Linear x3, ReLU x2; **B_triton**: triton_exp.fused_mlp2 x1, Linear x1

## 2 aten (torch.export)

| experiment | #ops | param inputs | ops |
|---|---|---|---|
| A_pytorch | 5 | 6 | `aten.linear` -> `aten.relu` -> `aten.linear` -> `aten.relu` -> `aten.linear` |
| B_triton | 2 | 6 | `triton_exp.fused_mlp2` -> `aten.linear` |

Op counts: **A_pytorch**: aten.linear x3, aten.relu x2; **B_triton**: triton_exp.fused_mlp2 x1, aten.linear x1

## 3 core aten

| experiment | #ops | param inputs | ops |
|---|---|---|---|
| A_pytorch | 8 | 6 | `aten.permute` -> `aten.addmm` -> `aten.relu` -> `aten.permute` -> `aten.addmm` -> `aten.relu` -> `aten.permute` -> `aten.addmm` |
| B_triton | 3 | 6 | `triton_exp.fused_mlp2` -> `aten.permute` -> `aten.addmm` |

Op counts: **A_pytorch**: aten.permute x3, aten.addmm x3, aten.relu x2; **B_triton**: triton_exp.fused_mlp2 x1, aten.permute x1, aten.addmm x1

## 4 inductor launches

| experiment | #ops | param inputs | ops |
|---|---|---|---|
| A_pytorch | 5 | 0 | `extern:mm` -> `triton_poi_fused_addmm_relu_0` -> `extern:mm` -> `triton_poi_fused_addmm_relu_0` -> `extern:addmm` |
| B_triton | 2 | 0 | `_mlp2_kernel_0` -> `extern:addmm` |

Op counts: **A_pytorch**: extern:mm x2, triton_poi_fused_addmm_relu_0 x2, extern:addmm x1; **B_triton**: _mlp2_kernel_0 x1, extern:addmm x1
