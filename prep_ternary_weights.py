
import torch
from transformers import AutoModelForCausalLM


MODEL_NAME = "EleutherAI/pythia-160m"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print(model)

LAYER_PATH = "gpt_neox.layers.0.mlp.dense_4h_to_h"

#Walk down to specific layer 
module = model
for attr in LAYER_PATH.split("."):
    module = getattr(module, attr)


w = module.weight.clone()
print(f"Grabbed '{LAYER_PATH}', shape={tuple(w.shape)}, dtype={w.dtype}")


def ternary_quantize(w: torch.Tensor):
    scale = w.abs().mean()
    w_scaled = w / scale
    w_ternary = torch.clamp(torch.round(w_scaled), -1, 1).to(torch.int8)
    return w_ternary, scale

w_ternary, scale = ternary_quantize(w)


# Sanity checks
print(f"Scale: {scale.item():.6f}")
print(f"Unique values: {torch.unique(w_ternary).tolist()}")
print(f"Fraction zero: {(w_ternary == 0).float().mean().item():.3f}")
 
w_reconstructed = w_ternary.float() * scale
rel_error = (w - w_reconstructed).norm() / w.norm()
print(f"Relative reconstruction error: {rel_error.item():.4f}")
 
torch.save({"w_ternary": w_ternary, "scale": scale, "shape": tuple(w.shape)}, "ternary_weight_sample.pt")
print("Saved to ternary_weight_sample.pt")
 