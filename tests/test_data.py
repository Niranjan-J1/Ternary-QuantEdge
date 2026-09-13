from datasets import load_dataset

# Streaming avoids downloading the whole dataset at once
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

# Peek at one example
example = next(iter(dataset))
print(example.keys())
print(example['text'][:500])