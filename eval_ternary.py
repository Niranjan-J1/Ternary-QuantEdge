from datasets import load_dataset
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantize import convert_to_ternary
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

MODEL_NAME = "EleutherAI/pythia-160m"
CHECKPOINT_PATH = "ternary_pythia160m_lr5e-05.pt"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Converting to ternary architecture...")
model = convert_to_ternary(model)
print("Loading trained checkpoint weights...")
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model = model.to("cuda")
model.eval()
print("Model ready.")

gc.collect()
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

print("Running LAMBADA eval...")
results = simple_evaluate(
    model=lm_obj,
    tasks=["lambada_openai"],
    limit=1000,
    bootstrap_iters=0
)
print(results["results"])