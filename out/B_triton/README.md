# B_triton: Triton kernel for fc1..relu2

Same MLP and weights, one output. `@triton_op triton_exp::fused_mlp2` does linear+relu+linear+relu in ONE kernel (4 PyTorch ops hidden in one node), then `fc3` is plain PyTorch. Compare with A_pytorch to see the Triton node replace those four ops.

| file | what |
|---|---|
| 1_fx_symbolic.txt/.svg/_readable.txt | `torch.fx.symbolic_trace`: one node per nn.Module / custom op |
| 2_aten_export.txt/.svg | `torch.export`: ATen ops, params as placeholders, shapes annotated |
| 3_core_aten.txt/.svg | after `run_decompositions()`: linear -> permute + addmm |
| 4_inductor_code.py / 4_inductor_fused.svg | Inductor output: kernel launches + fused-op diagram |
