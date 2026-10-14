
import torch

# Load saved embeddings
data = torch.load("train_layer2.pt", weights_only=True)

# Check dimensions
print("Embeddings shape:", data["embeddings"].shape)
print("Labels shape:", data["labels"].shape)

# Show the first embedding
print("\nFirst embedding:")
print(data["embeddings"][0])

# Show the first 10 POS labels
print("\nFirst 10 labels:")
print(data["labels"][:10])
