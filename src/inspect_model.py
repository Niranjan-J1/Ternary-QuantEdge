from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
print(model)