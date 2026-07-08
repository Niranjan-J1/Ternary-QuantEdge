from transformers import AutoModelForCausalLM
from src.quantize import convert_to_ternary, TernaryLinear

model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
model = convert_to_ternary(model)

# Check the conversion worked
first_layer = model.gpt_neox.layers[0]
print("query_key_value type:", type(first_layer.attention.query_key_value))
print("dense type:", type(first_layer.attention.dense))
print("dense_h_to_4h type:", type(first_layer.mlp.dense_h_to_4h))
print("dense_4h_to_h type:", type(first_layer.mlp.dense_4h_to_h))

# Confirm embed_in/embed_out were NOT touched
print("embed_out type (should still be nn.Linear):", type(model.embed_out))