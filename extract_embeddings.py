
import torch
from test_bert import dataset, tokenizer, model, device

model.eval()

POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET",
    "INTJ", "NOUN", "NUM", "PART", "PRON",
    "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]

pos_to_id = {
    tag: i for i, tag in enumerate(POS_TAGS)
}


def extract_embeddings(data, layer):

    embeddings = []
    labels = []

    for sentence in data:

        sentence_embeddings = []
        sentence_labels = []
        words = []
        pos_tags = []

        for token in sentence:
            if isinstance(token["id"], int):
                words.append(token["form"])
                pos_tags.append(token["upos"])

        inputs = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt"
        )

        word_ids = inputs.word_ids(batch_index=0)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[layer][0]

        for word_index in range(len(words)):

            subtoken_vectors = []

            for token_index, word_id in enumerate(word_ids):
                if word_id == word_index:
                    subtoken_vectors.append(hidden_states[token_index])

            if len(subtoken_vectors) > 0:
                word_vector = torch.stack(subtoken_vectors).mean(dim=0)

                sentence_embeddings.append(word_vector.cpu())
                sentence_labels.append(pos_to_id[pos_tags[word_index]])

        embeddings.append(torch.stack(sentence_embeddings))
        labels.append(sentence_labels)

    X = torch.cat(embeddings, dim=0)

    y = torch.tensor(
        [label for sentence in labels for label in sentence],
        dtype=torch.long
    )

    return X, y


print("Extracting training embeddings...")
X_dat, y_dat = extract_embeddings(dataset["train"], 8)

print("Extracting validation embeddings...")
X_val, y_val = extract_embeddings(dataset["validation"], 8)

print("Extracting test embeddings...")
X_test, y_test = extract_embeddings(dataset["test"], 8)


print("Train:", X_dat.shape, y_dat.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


torch.save({"embeddings": X_dat, "labels": y_dat}, "train_layer8.pt")
torch.save({"embeddings": X_val, "labels": y_val}, "validation_layer8.pt")
torch.save({"embeddings": X_test, "labels": y_test}, "test_layer8.pt")
