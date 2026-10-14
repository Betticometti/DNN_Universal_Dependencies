import os

from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(42)

# EPOCHS -- Number of complete passes over the training dataset.
#           Too few epochs may lead to underfitting.
#           Too many epochs may lead to overfitting.
#           Try 5, 10, or 20 epochs and compare validation accuracy.

EPOCHS = 10

# LR -- Learning rate for the Adam optimizer.
#       A larger learning rate may speed up training but make it unstable.
#       A smaller learning rate gives smaller updates but may require
#       more epochs.
#       Consider 1e-2, 1e-3, or 1e-4.

LR = 1e-3

# BATCH_SIZE -- Number of word embeddings used per parameter update.
#               Smaller batches give more frequent, noisier updates.
#               Larger batches give fewer updates per epoch.
#               Consider 32, 64, 128, or 256.

BATCH_SIZE = 128

# -------------------------------------------------------------------------------


# Linear probe for POS classification
class Probe(nn.Module):
    #768 e 17
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Evaluate classification accuracy
def evaluate_probe(probe, loader, device):
    probe.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = probe(x)
            predictions = outputs.argmax(1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total


# Train the probe using the training and validation datasets
def train_probe(X_train, y_train, X_val, y_val, num_classes):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    probe = Probe(X_train.shape[1], num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)

    best_val = -1.0
    best_state = None

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        probe.train()

        correct = 0
        total = 0
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = probe(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        val_acc = evaluate_probe(probe, val_loader, device)

        train_losses.append(total_loss / total)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                k: v.detach().clone()
                for k, v in probe.state_dict().items()
            }

        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"Loss: {total_loss/total:.4f} "
              f"Train accuracy: {train_acc:.4f} "
              f"Validation accuracy: {val_acc:.4f}")

    probe.load_state_dict(best_state)

    print(f"Best validation accuracy: {best_val:.4f}")
    plot_training(train_losses, train_accuracies, val_accuracies)

    # Save the trained probe
    # torch.save(probe.state_dict(), "probe_layer12.pt")

    return probe


# Plot training results
def plot_training(train_losses, train_accuracies, val_accuracies):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training accuracy")
    plt.plot(epochs, val_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig("plots/probe_layer1_training.png")
    plt.close()



# Show incorrectly classified POS tags

# Show incorrectly classified POS tags and evaluation metrics
def show_errors(probe, X_val, y_val, id_to_pos, max_errors=30):

    device = next(probe.parameters()).device
    probe.eval()

    errors = []
    predictions = []
    true_labels = []

    with torch.no_grad():
        for i in range(len(X_val)):
            x = X_val[i].unsqueeze(0).to(device)

            prediction = probe(x).argmax(1).item()
            true_label = y_val[i].item()

            predictions.append(prediction)
            true_labels.append(true_label)

            if prediction != true_label:
                errors.append((i, true_label, prediction))

    # Evaluation metrics
    accuracy = 1 - len(errors) / len(y_val)
    macro_f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("PROBE EVALUATION - VALIDATION SET")
    print("=" * 60)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION ERRORS - VALIDATION SET")
    print("=" * 60)

    print(f"Total errors: {len(errors)}")
    print(f"Total words: {len(y_val)}")

    print(f"\nFirst {min(max_errors, len(errors))} errors:")
    print(f"{'Index':<10}{'True POS':<15}{'Predicted POS':<15}")
    print("-" * 40)

    for i, true_label, prediction in errors[:max_errors]:
        print(f"{i:<10}{id_to_pos[true_label]:<15}{id_to_pos[prediction]:<15}")




POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET",
    "INTJ", "NOUN", "NUM", "PART", "PRON",
    "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]

id_to_pos = {i: tag for i, tag in enumerate(POS_TAGS)}



# Load layer 12 embeddings
train_data = torch.load("embeddings/train_layer1.pt", weights_only=True)
val_data = torch.load("embeddings/validation_layer1.pt", weights_only=True)

X_train = train_data["embeddings"]
y_train = train_data["labels"]

X_val = val_data["embeddings"]
y_val = val_data["labels"]

# Train the probe
probe = train_probe(X_train, y_train, X_val, y_val, 17)

# Analyze classification errors on validation set
show_errors(probe, X_val, y_val, id_to_pos)