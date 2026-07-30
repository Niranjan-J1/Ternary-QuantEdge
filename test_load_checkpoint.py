from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantize import convert_to_ternary

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
print("Converting to ternary...")
model = convert_to_ternary(model)
print("Loading checkpoint...")
model.load_state_dict(torch.load('ternary_pythia160m.pt'))
print("Moving to GPU...")
model = model.to("cuda")
model.eval()
print("SUCCESS - model loaded fully.")