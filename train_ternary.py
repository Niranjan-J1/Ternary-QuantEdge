from datasets import load_dataset
import torch
import sys
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from src.quantize import convert_to_ternary

print("Imports done.")

# --- Config ---
MODEL_NAME = "EleutherAI/pythia-160m"
MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-5
NUM_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
LOG_EVERY = 50

print(f"Learning rate: {LEARNING_RATE}, Steps: {NUM_STEPS}")

# --- Setup ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Model loaded (CPU). Converting to ternary...")
model = convert_to_ternary(model)
print("Converted to ternary. Moving to GPU...")
model = model.to("cuda")
model.train()
print("Model on GPU, in train mode.")

print("Loading dataset (streaming)...")
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
print("Dataset ready.")

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=MAX_LENGTH)

print("Mapping tokenizer over dataset...")
tokenized = dataset.map(tokenize_fn, remove_columns=["text", "meta"])
print("Tokenized dataset ready.")

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
loader = DataLoader(tokenized, batch_size=BATCH_SIZE, collate_fn=collator)
print("DataLoader ready.")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
print("Optimizer ready. Starting training loop...")

# --- Training loop ---
step = 0
for batch in loader:
    batch = {k: v.to("cuda") for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % LOG_EVERY == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

    step += 1
    if step >= NUM_STEPS:
        break

# --- Save checkpoint ---
checkpoint_name = f"ternary_pythia160m_lr{LEARNING_RATE}.pt"
torch.save(model.state_dict(), checkpoint_name)
print(f"Training complete. Checkpoint saved as {checkpoint_name}.")