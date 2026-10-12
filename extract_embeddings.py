import torch
from test_bert import dataset, tokenizer, model, device

def extract_embeddings(data, layer):

    embeddings = []

    for sentence in data:

        words = []

        for token in sentence:
            if isinstance(token["id"], int):
                words.append(token["form"])

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

                embeddings.append(word_vector.cpu())

    return torch.stack(embeddings)


# Test on the first training sentence
X = extract_embeddings(
    dataset["train"][:1],
    6
)

print("Embeddings shape:", X.shape)
print("First word embedding:", X[0])