"""Two separate models over the SAME 3-layer MLP (fc1 -> relu -> fc2 -> relu -> fc3), same weights, one output each.

  A_pytorch   no Triton: all five layers are ordinary PyTorch modules.
  B_triton    Triton: fused_mlp2 (fc1+relu+fc2+relu in ONE @triton_op kernel), then fc3 in PyTorch.
"""
import torch
import torch.nn as nn

import kernels  # noqa: F401  (registers torch.ops.triton_exp.*)

IN, HID, OUT = 16, 32, 8


class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.act1 = nn.Linear(IN, HID), nn.ReLU()
        self.fc2, self.act2 = nn.Linear(HID, HID), nn.ReLU()
        self.fc3 = nn.Linear(HID, OUT)

    def forward(self, x):
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))


class TritonMLP3(MLP3):
    def forward(self, x):
        h = torch.ops.triton_exp.fused_mlp2(x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias)
        return self.fc3(h)


def build_all(device="cuda"):
    torch.manual_seed(0)
    a = MLP3().to(device).eval()
    b = TritonMLP3().to(device).eval()
    b.load_state_dict(a.state_dict())
    return {"A_pytorch": a, "B_triton": b}
