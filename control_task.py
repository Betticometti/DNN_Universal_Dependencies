
import os
import random
import torch
from collections import Counter
from conllu import parse_incr

LAYER_NUMBER = 12

os.makedirs("control_embeddings", exist_ok=True)

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

POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET",
    "INTJ", "NOUN", "NUM", "PART", "PRON",
    "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]

pos_to_id = {tag: i for i, tag in enumerate(POS_TAGS)}

DICTIONARY_PATH = "control_embeddings/random_dictionary.pt"


# Create random POS labels for each word
def create_random_dictionary():

    if os.path.exists(DICTIONARY_PATH):
        print("Random dictionary already exists.")
        return

    random.seed(42)

    # Count POS tags in the training set
    pos_counts = Counter()

    for sentence in dataset["train"]:
        for token in sentence:
            if isinstance(token["id"], int):
                pos_counts[token["upos"]] += 1

    weights = [pos_counts[tag] for tag in POS_TAGS]

    # Assign random labels to each word
    random_dictionary = {}

    for split in dataset:
        for sentence in dataset[split]:
            for token in sentence:
                if isinstance(token["id"], int):
                    word = token["form"].lower()

                    if word not in random_dictionary:
                        random_dictionary[word] = random.choices(
                            range(len(POS_TAGS)),
                            weights=weights,
                            k=1
                        )[0]

    torch.save(random_dictionary, DICTIONARY_PATH)

    print("Random dictionary saved.")
    print("Unique words:", len(random_dictionary))


# Create a control task file using the saved dictionary
def create_control_file(file_path):

    random_dictionary = torch.load(
        DICTIONARY_PATH, weights_only=True
    )

    file_name = os.path.basename(file_path)
    split = file_name.split("_")[0]

    if split not in dataset:
        raise ValueError("Invalid dataset split.")

    data = torch.load(file_path, weights_only=True)

    random_labels = []
    original_labels = []

    for sentence in dataset[split]:
        for token in sentence:
            if isinstance(token["id"], int):
                word = token["form"].lower()

                random_labels.append(random_dictionary[word])
                original_labels.append(pos_to_id[token["upos"]])

    random_labels = torch.tensor(random_labels, dtype=torch.long)
    original_labels = torch.tensor(original_labels, dtype=torch.long)

    # Check that labels match the original embeddings
    assert len(random_labels) == len(data["embeddings"])
    assert torch.equal(original_labels, data["labels"])

    # Save embeddings with random labels
    control_data = {
        "embeddings": data["embeddings"],
        "labels": random_labels
    }

    output_path = os.path.join("control_embeddings", file_name)
    torch.save(control_data, output_path)

    print("Control file saved:", output_path)


# Run
create_random_dictionary()

create_control_file(f"embeddings/train_layer{LAYER_NUMBER}.pt")
create_control_file(f"embeddings/validation_layer{LAYER_NUMBER}.pt")
create_control_file(f"embeddings/test_layer{LAYER_NUMBER}.pt")
