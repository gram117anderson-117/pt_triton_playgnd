"""Run every experiment and write, per experiment, into out/<exp>/:

  *.txt           raw FX IR, i.e. str(gm.graph); *_readable.txt = same graph printed as python code
  1_fx_symbolic   torch.fx.symbolic_trace  (module-level FX graph)
  2_aten_export   torch.export             (ATen-op graph; "aten" output)
  3_core_aten     ep.run_decompositions()  (Core ATen, asserted: every non-triton op is an aten op tagged `core`)
  4_inductor_code.py / .svg   torch.compile / Inductor (generated kernels + fused-kernel diagram)
  README.md                   what the experiment is

and out/COMPARISON.md comparing all experiments stage by stage.

Run:  .venv/bin/python run_experiments.py      (needs graphviz `dot` for the .svg files)
"""
import collections
import glob
import os
import re
import shutil

import torch
from torch._inductor.utils import fresh_cache
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.fx.passes.shape_prop import ShapeProp

from models import build_all

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
x = torch.randn(4, 16, device="cuda")

EXPERIMENTS = {
    "A_pytorch": ("Pure PyTorch 3-layer MLP",
                  "fc1 -> relu -> fc2 -> relu -> fc3, one output. No Triton. The reference."),
    "B_triton": ("Triton kernel for fc1..relu2",
                 "Same MLP and weights, one output. `@triton_op triton_exp::fused_mlp2` does "
                 "linear+relu+linear+relu in ONE kernel (4 PyTorch ops hidden in one node), then `fc3` is plain "
                 "PyTorch. Compare with A_pytorch to see the Triton node replace those four ops."),
}


def svg(gm, path):
    FxGraphDrawer(gm, os.path.basename(path), ignore_getattr=False).get_dot_graph().write_svg(path)


def raw(gm, base):
    """<base>.txt = raw FX IR (str(graph)); <base>_readable.txt = same graph as python code."""
    open(base + ".txt", "w").write(str(gm.graph) + "\n")
    open(base + "_readable.txt", "w").write(gm.print_readable(print_output=False))


def op_list(gm):
    """Ordered compute ops of an FX graph (placeholders/output skipped), plus param-input count."""
    ops, nparams = [], 0
    for n in gm.graph.nodes:
        if n.op in ("placeholder", "output"):
            nparams += n.op == "placeholder" and n.name.startswith("p_")
        elif n.op == "call_module":
            ops.append(type(gm.get_submodule(n.target)).__name__)
        elif n.op == "get_attr":
            nparams += 1
        else:
            ops.append(re.sub(r"\.default$", "", str(n.target).replace("torch.ops.", "")))
    return ops, nparams


def check_aten_level(gm):
    """After decomposition every call_function must be an ATen op carrying the `core` tag, except triton_exp.*."""
    bad = []
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        t = n.target
        if t.namespace == "triton_exp":
            continue
        if t.namespace != "aten" or torch.Tag.core not in t.tags:
            bad.append(str(t))
    assert not bad, f"non-core-ATen ops remain after run_decompositions: {bad}"


def run(name, model):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    title, desc = EXPERIMENTS[name]
    summary = {}

    # Stage 1: FX symbolic trace (module level)
    gm = torch.fx.symbolic_trace(model)
    raw(gm, f"{d}/1_fx_symbolic")  # dump BEFORE ShapeProp so the text is untouched
    ShapeProp(gm).propagate(x)  # only to annotate the .svg with shapes
    svg(gm, f"{d}/1_fx_symbolic.svg")
    summary["1 fx symbolic_trace"] = op_list(gm)

    # Stage 2: torch.export -> ATen
    ep = torch.export.export(model, (x,))
    raw(ep.graph_module, f"{d}/2_aten_export")
    svg(ep.graph_module, f"{d}/2_aten_export.svg")
    summary["2 aten (torch.export)"] = op_list(ep.graph_module)

    # Stage 3: Core ATen
    core = ep.run_decompositions()
    raw(core.graph_module, f"{d}/3_core_aten")
    svg(core.graph_module, f"{d}/3_core_aten.svg")
    check_aten_level(core.graph_module)
    summary["3 core aten"] = op_list(core.graph_module)

    # Stage 4: Inductor
    import torch._inductor.config as ic
    import torch._dynamo.config as dc
    dbg = os.path.join(d, "_inductor_debug")
    shutil.rmtree(dbg, ignore_errors=True)
    torch._dynamo.reset()
    dc.debug_dir_root, ic.trace.enabled, ic.trace.graph_diagram = dbg, True, True
    with fresh_cache(), torch.no_grad():
        torch.compile(model, fullgraph=True)(x)
    ic.trace.enabled = False
    code = glob.glob(f"{dbg}/**/output_code.py", recursive=True)[0]
    shutil.copy(code, f"{d}/4_inductor_code.py")
    shutil.copy(os.path.join(os.path.dirname(code), "graph_diagram.svg"), f"{d}/4_inductor_fused.svg")
    shutil.rmtree(dbg)
    src = open(f"{d}/4_inductor_code.py").read()
    call = src.split("def call(")[1].split("def benchmark")[0]
    kernels = [m.group(1) and f"extern:{m.group(1)}" or m.group(2)
               for m in re.finditer(r"extern_kernels\.(\w+)\(|(\w+)\.run\(", call)]
    summary["4 inductor launches"] = (kernels, 0)

    open(f"{d}/README.md", "w").write(
        f"# {name}: {title}\n\n{desc}\n\n| file | what |\n|---|---|\n"
        "| 1_fx_symbolic.txt/.svg/_readable.txt | `torch.fx.symbolic_trace`: one node per nn.Module / custom op |\n"
        "| 2_aten_export.txt/.svg | `torch.export`: ATen ops, params as placeholders, shapes annotated |\n"
        "| 3_core_aten.txt/.svg | after `run_decompositions()`: linear -> permute + addmm |\n"
        "| 4_inductor_code.py / 4_inductor_fused.svg | Inductor output: kernel launches + fused-op diagram |\n")
    print("done", name)
    return summary


def comparison(results):
    names = list(results)
    L = ["# Comparison\n"]
    for n in names:
        L.append(f"- **{n}** - {EXPERIMENTS[n][0]}: {EXPERIMENTS[n][1]}")
    L.append("\nOutputs match across experiments (checked in `main`). Cells list compute ops in graph order.\n")
    for stage in results[names[0]]:
        L.append(f"\n## {stage}\n")
        L.append("| experiment | #ops | param inputs | ops |\n|---|---|---|---|")
        for n in names:
            ops, p = results[n][stage]
            L.append(f"| {n} | {len(ops)} | {p} | " + " -> ".join(f"`{o}`" for o in ops) + " |")
        L.append("\nOp counts: " + "; ".join(
            f"**{n}**: " + ", ".join(f"{o} x{c}" for o, c in collections.Counter(results[n][stage][0]).items())
            for n in names))
    open(f"{OUT}/COMPARISON.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    assert shutil.which("dot"), "graphviz `dot` not found (sudo apt install graphviz)"
    shutil.rmtree(OUT, ignore_errors=True)
    models = build_all()
    ref = models["A_pytorch"](x)
    for n, m in models.items():
        outs = m(x)
        for o in (outs if isinstance(outs, tuple) else (outs,)):
            assert torch.allclose(o, ref, atol=1e-5), f"{n} output differs from baseline"
    print("all outputs match the pure-PyTorch reference")
    comparison({n: run(n, m) for n, m in models.items()})
