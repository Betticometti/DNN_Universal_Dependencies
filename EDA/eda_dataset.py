
from collections import Counter
from pathlib import Path
from conllu import parse_incr

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILES = {
    "Train": DATA_DIR / "en_ewt-ud-train.conllu",
    "Validation": DATA_DIR / "en_ewt-ud-dev.conllu",
    "Test": DATA_DIR / "en_ewt-ud-test.conllu",
}


def analyze_dataset(path):
    """Count sentences, actual words, and universal POS tags in a CoNLL-U file."""
    sentence_count = 0
    word_count = 0
    pos_counts = Counter()

    with path.open("r", encoding="utf-8") as file:
        for sentence in parse_incr(file):
            sentence_count += 1
            for token in sentence:
                # Skip multiword-token ranges and empty nodes.
                if not isinstance(token["id"], int):
                    continue
                word_count += 1
                if token["upos"] is not None:
                    pos_counts[token["upos"]] += 1

    return sentence_count, word_count, pos_counts


def main():
    
    results = {}
    for split, path in DATA_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        results[split] = analyze_dataset(path)

    print("\n1. DATASET SIZE")
    print("-" * 60)
    print(f"{'Split':<15}{'Sentences':>15}{'Words':>15}")
    for split, (sentences, words, _) in results.items():
        print(f"{split:<15}{sentences:>15,}{words:>15,}")

    train_counts = results["Train"][2]
    print("\n2. UNIVERSAL POS TAGS (TRAINING SET)")
    print("-" * 60)
    print(f"Number of POS tags: {len(train_counts)}")
    print("POS tags:", ", ".join(sorted(train_counts)))

    print("\n3. POS TAG DISTRIBUTION (TRAINING SET)")
    print("-" * 60)
    print(f"{'POS tag':<15}{'Count':>15}{'Percentage':>15}")
    total = sum(train_counts.values())
    for tag, count in train_counts.most_common():
        percentage = 100 * count / total if total else 0
        print(f"{tag:<15}{count:>15,}{percentage:>14.2f}%")


if __name__ == "__main__":
    main()
