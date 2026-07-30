from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize_fn, remove_columns=["text", "meta"])

example = next(iter(tokenized))
print(example.keys())
print("Token count:", len(example["input_ids"]))
print("First 20 tokens:", example["input_ids"][:20])