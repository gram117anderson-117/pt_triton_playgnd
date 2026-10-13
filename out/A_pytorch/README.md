# A_pytorch: Pure PyTorch 3-layer MLP

fc1 -> relu -> fc2 -> relu -> fc3, one output. No Triton. The reference.

| file | what |
|---|---|
| 1_fx_symbolic.txt/.svg/_readable.txt | `torch.fx.symbolic_trace`: one node per nn.Module / custom op |
| 2_aten_export.txt/.svg | `torch.export`: ATen ops, params as placeholders, shapes annotated |
| 3_core_aten.txt/.svg | after `run_decompositions()`: linear -> permute + addmm |
| 4_inductor_code.py / 4_inductor_fused.svg | Inductor output: kernel launches + fused-op diagram |
