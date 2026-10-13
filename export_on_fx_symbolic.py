"""torch.export.export() on each model, two ways:

  (1) module    : export the original nn.Module
  (2) fx_symbolic: export the GraphModule produced by torch.fx.symbolic_trace(model)

Prints the full ExportedProgram (graph + signature) and writes it to out/export_on_fx_symbolic/.
Run:  .venv/bin/python export_on_fx_symbolic.py
"""
import os

import torch

from models import build_all

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "export_on_fx_symbolic")
os.makedirs(D, exist_ok=True)
x = torch.randn(4, 16, device="cuda")

for name, model in build_all().items():
    gm = torch.fx.symbolic_trace(model)
    for how, target in [("module", model), ("fx_symbolic", gm)]:
        title = f"{name} :: export({how})"
        try:
            ep = torch.export.export(target, (x,))
            text = str(ep)
            ok = torch.allclose(ep.module()(x), model(x), atol=1e-5)
            text += f"\n\n# ep.module()(x) matches eager model: {ok}\n"
        except Exception as e:
            text = f"FAILED: {type(e).__name__}: {str(e)[:2000]}"
        open(f"{D}/{name}__{how}.txt", "w").write(text)
        print(f"\n{'=' * 90}\n{title}\n{'=' * 90}\n{text}")
