from transformers import AutoModelForCausalLM
from src.quantize import convert_to_ternary, TernaryLinear

model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
model = convert_to_ternary(model)

all_correct = True
for i, layer in enumerate(model.gpt_neox.layers):
    checks = [
        isinstance(layer.attention.query_key_value, TernaryLinear),
        isinstance(layer.attention.dense, TernaryLinear),
        isinstance(layer.mlp.dense_h_to_4h, TernaryLinear),
        isinstance(layer.mlp.dense_4h_to_h, TernaryLinear),
    ]
    if not all(checks):
        print(f"Layer {i} FAILED conversion:", checks)
        all_correct = False

print("All 12 layers correctly converted:" if all_correct else "SOME LAYERS FAILED", all_correct)