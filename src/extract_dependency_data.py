import torch
from test_bert import dataset
import os

os.makedirs("dependency_labels", exist_ok=True)

# Estrae le etichette 'head' e le lunghezze delle frasi
def extract_dependencies(data):
    labels = []
    lengths = []

    for sentence in data:
        sentence_labels = []
        
        for token in sentence:
            # Stesso filtro per gli embedding e i POS tag
            if isinstance(token["id"], int):
                # estraiamo indice del padre (0 indica il ROOT)
                sentence_labels.append(int(token["head"]))
        
        if len(sentence_labels) > 0:
            labels.append(sentence_labels)
            lengths.append(len(sentence_labels))

    # appiattiamo le etichette in un singolo tensore esattamente come per la variabile y dei POS tag
    y = torch.tensor(
        [label for sentence in labels for label in sentence],
        dtype=torch.long
    )

    return y, lengths

print("--- Starting extraction ---")

for split in ["train", "validation", "test"]:
    print(f"\nElaborazione set: {split}...")
    
    y_deps, lengths = extract_dependencies(dataset[split])
    
    print(f"Shape labels {split}: {y_deps.shape}")
    print(f"Total sentences: {len(lengths)}")
    print(f"Total words: {sum(lengths)}")
    
    torch.save({
        "heads": y_deps, 
        "lengths": lengths
    }, f"dependency_labels/{split}_deps.pt")

print("\nExtraction complete")