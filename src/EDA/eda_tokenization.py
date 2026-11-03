
from pathlib import Path
from collections import Counter
from conllu import parse_incr
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATA_FILES = {
    "Train": DATA_DIR / "en_ewt-ud-train.conllu",
    "Validation": DATA_DIR / "en_ewt-ud-dev.conllu",
    "Test": DATA_DIR / "en_ewt-ud-test.conllu",
}

# Load sentences from a CoNLL-U dataset file
def load_dataset(path):
    with path.open("r", encoding="utf-8") as file:
        return list(parse_incr(file))

# Count words split into subtokens and analyze their POS tags
def analyze_subwords(data):
    total_words = 0
    split_words = []
    split_pos = Counter()

    for sentence in data:
        for token in sentence:
            if not isinstance(token["id"], int):
                continue

            word = token["form"]
            subtokens = tokenizer.tokenize(word)
            total_words += 1

            if len(subtokens) > 1:
                split_words.append((word, subtokens))
                split_pos[token["upos"]] += 1

    print("\n1. SUBWORD ANALYSIS (TRAINING SET)")
    print("-" * 60)
    print("Total words:", total_words)
    print("Words split into subwords:", len(split_words))
    print(f"Percentage: {len(split_words) / total_words * 100:.2f}%")

    print("\nExamples of split words:")
    for word, subtokens in split_words[:20]:
        print(f"{word:<20} -> {subtokens}")

    print("\nPOS tags of split words:")
    for tag, count in split_pos.most_common():
        print(f"{tag:<10} {count:>6}")

# Check tokenized sentence lengths and identify sequences exceeding 512 tokens
def analyze_sentence_lengths(data, name):
    max_tokens = 0
    long_sentences = []

    for i, sentence in enumerate(data):
        words = [
            token["form"]
            for token in sentence
            if isinstance(token["id"], int)
        ]

        inputs = tokenizer(words, is_split_into_words=True)
        num_tokens = len(inputs["input_ids"])

        max_tokens = max(max_tokens, num_tokens)

        if num_tokens > 512:
            long_sentences.append((i, num_tokens))

    print(f"\n{name.upper()} SENTENCE LENGTHS")
    print("-" * 60)
    print("Total sentences:", len(data))
    print("Sentences exceeding 512 tokens:", len(long_sentences))
    print("Maximum sentence length:", max_tokens)
    print("Long sentences:", long_sentences)

# Run subword and sentence length analysis on the dataset
def main():
    datasets = {
        name: load_dataset(path)
        for name, path in DATA_FILES.items()
    }

    analyze_subwords(datasets["Train"])

    print("\n2. SENTENCE LENGTH ANALYSIS")
    for name, data in datasets.items():
        analyze_sentence_lengths(data, name)


if __name__ == "__main__":
    main()
