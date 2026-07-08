import torch
import torch.nn as nn
import torch.nn.functional as F

def ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to {-1, 0, +1} with a per-tensor scale factor.
    Returns (quantized_weights, scale).
    """
    # Step 1: compute the scale factor
    scale = w.abs().mean()
    
    # Step 2: normalize weights by scale, then round to nearest of {-1, 0, +1}
    w_scaled = w / (scale + 1e-8)
    w_ternary = torch.clamp(torch.round(w_scaled), -1, 1)
    
    return w_ternary, scale

class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        w_ternary, scale = ternary_quantize(w)
        return w_ternary * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ternary_ste(w: torch.Tensor) -> torch.Tensor:
    return TernarySTE.apply(w)


class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    @classmethod
    def from_linear(cls, linear: nn.Linear):
        """Build a TernaryLinear that copies weights from an existing nn.Linear."""
        new_layer = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        with torch.no_grad():
            new_layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                new_layer.bias.copy_(linear.bias)
        return new_layer

    def forward(self, x):
        w_q = ternary_ste(self.weight)
        return F.linear(x, w_q, self.bias)