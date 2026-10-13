import torch
from test_bert import dataset, tokenizer, model, device

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
                sentence_labels.append(pos_tags[word_index])

        embeddings.append(torch.stack(sentence_embeddings))
        labels.append(sentence_labels)

    return embeddings, labels


# Test on the first three training sentences
X, y = extract_embeddings(
    dataset["train"][:3],
    2
)

print("Word:", dataset["train"][2][1]["form"])
print("Label:", y[2][1])
print("Embedding:", X[2][1])