
import torch

# Load saved embeddings
#data = torch.load("embeddings/train_layer9.pt", weights_only=True)
data = torch.load("control_embeddings/train_layer2.pt", weights_only=True)

# Show information about the loaded data
print("Loaded embeddings from file: control_embeddings/train_layer2.pt")

# Show the first embedding
print("\nFirst embedding:")
print(data["embeddings"][0])

# Show the first 10 POS labels
print("\nFirst 10 labels:")
print(data["labels"][:10])
