import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from transformers import AutoTokenizer, AutoModel
from conllu import parse_incr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
def load_data(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(parse_incr(file))

data_files = {
    "train": "data/en_ewt-ud-train.conllu",
    "validation": "data/en_ewt-ud-dev.conllu",
    "test": "data/en_ewt-ud-test.conllu"
}

dataset = {
    split: load_data(path)
    for split, path in data_files.items()
}

# Check dataset
print("Train sentences:", len(dataset["train"]))
print("Validation sentences:", len(dataset["validation"]))
print("Test sentences:", len(dataset["test"]))

print([token["form"] for token in dataset["train"][0]
       if isinstance(token["id"], int)])


# Load pretrained BERT-base-uncased
model_name = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.to(device) # directly on gpu

# Freeze BERT parameters
model.eval()
for param in model.parameters():
    param.requires_grad = False
