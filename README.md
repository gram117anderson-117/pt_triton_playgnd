# pt_triton_playgnd

A small playground for seeing how a custom Triton kernel shows up at each stage of the PyTorch compile stack: FX, `torch.export`, Core ATen, and Inductor.

## Experiments

Both models are the same 3-layer MLP (`fc1 -> relu -> fc2 -> relu -> fc3`) with the same weights:

- **A_pytorch**: plain PyTorch. This is the reference.
- **B_triton**: `fc1 + relu + fc2 + relu` run as one Triton kernel, registered with `@triton_op` as `torch.ops.triton_exp.fused_mlp2`, followed by `fc3` in PyTorch.

Every stage keeps the Triton op as a single opaque node, and Inductor emits the user kernel unchanged:

| stage | A_pytorch | B_triton |
|---|---|---|
| FX `symbolic_trace` | 5 ops | 2 ops |
| `torch.export` (ATen) | 5 ops | 2 ops |
| Core ATen | 8 ops | 3 ops |
| Inductor launches | 5 | 2 (`_mlp2_kernel` + `addmm`) |

See [out/COMPARISON.md](out/COMPARISON.md) for the full breakdown.

## Files

| file | what |
|---|---|
| `kernels.py` | Triton kernel + `@triton_op` / `wrap_triton` registration |
| `models.py` | the two models (`build_all()`) |
| `run_experiments.py` | dumps every stage per model to `out/<exp>/` (raw IR, readable code, SVG graphs) and writes `out/COMPARISON.md` |
| `export_on_fx_symbolic.py` | compares `torch.export` on the original module with `torch.export` on the `symbolic_trace` GraphModule; writes `out/export_on_fx_symbolic/` |

## Running

Needs a CUDA GPU. Tested with torch 2.14.1+cu130 and triton 3.8.0. The `.svg` files also need graphviz (`dot`).

```bash
.venv/bin/python run_experiments.py
.venv/bin/python export_on_fx_symbolic.py
```
