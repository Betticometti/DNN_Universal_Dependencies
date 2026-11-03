import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

# Same configuration as probe.py
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 128
NUM_LAYERS = 12

torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "scalar_mixing"

POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET",
    "INTJ", "NOUN", "NUM", "PART", "PRON",
    "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

id_to_pos = {i: tag for i, tag in enumerate(POS_TAGS)}


# Load and combine embeddings from all 12 BERT layers
def load_embeddings(split, data_dir):
    X = None
    labels = None

    for layer in range(1, NUM_LAYERS + 1):
        file_path = data_dir / f"{split}_layer{layer}.pt"
        data = torch.load(file_path, map_location="cpu", weights_only=True)
        x = data["embeddings"]
        y = data["labels"].long()

        if X is None:
            labels = y.clone()
            X = torch.empty((len(y), NUM_LAYERS, x.shape[1]), dtype=torch.float32)
        else:
            if not torch.equal(labels, y):
                raise ValueError(f"Labels are different across layers in {file_path}")
        if x.ndim != 2 or x.shape != (len(labels), 768):
            raise ValueError(f"Unexpected embedding shape in {file_path}: {tuple(x.shape)}")

        X[:, layer - 1, :] = x
        print(f"Loaded {split} layer {layer}/{NUM_LAYERS}")
        del data, x, y

    return X, labels

# Linear probe that learns how to combine embeddings from different BERT layers
class ScalarMixProbe(nn.Module):
    # Initialize the learnable layer weights and linear classifier
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(NUM_LAYERS))
        self.model = nn.Linear(input_dim, num_classes)
    
    # Combine layer embeddings using learned weights and predict POS labels
    def forward(self, x):
        weights = torch.softmax(self.layer_weights, dim=0)
        mixed = (x * weights.view(1, NUM_LAYERS, 1)).sum(dim=1)
        return self.model(mixed)

# Calculate classification accuracy on a given dataset
def evaluate_probe(probe, loader, device):
    probe.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predictions = probe(x).argmax(1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
    return correct / total

# Plot and save training loss, training accuracy, and validation accuracy
def plot_training(train_losses, train_accuracies, val_accuracies, out_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss (scalar mixing + linear)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training accuracy")
    plt.plot(epochs, val_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "training.png")
    plt.close()

# Print and save the learned weights of all 12 BERT layers
def save_layer_weights(probe, out_dir):
    weights = torch.softmax(probe.layer_weights.detach().cpu(), dim=0)
    layers = torch.arange(1, NUM_LAYERS + 1)
    cog = (layers * weights).sum().item()

    print(f"Center of Gravity: {cog:.4f}")
    with open(out_dir / "layer_weights.txt", "w") as f:
        for layer, weight in enumerate(weights.tolist(), start=1):
            line = f"Layer {layer:2d}: {weight:.6f}"
            print(line)
            f.write(line + "\n")
        f.write(f"Highest weight: Layer {weights.argmax().item() + 1}\n")
        f.write(f"Center of Gravity: {cog:.4f}\n")

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, NUM_LAYERS + 1), weights.tolist())
    plt.xticks(range(1, NUM_LAYERS + 1))
    plt.xlabel("BERT layer")
    plt.ylabel("Softmax weight")
    plt.title("Scalar Mixing - Learned Layer Weights")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "layer_weights.png")
    plt.close()

# Train the probe and restore the model with the best validation accuracy
def train_probe(X_train, y_train, X_val, y_val, num_classes, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    probe = ScalarMixProbe(X_train.shape[2], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)

    best_val = -1.0
    best_epoch = 0
    best_state = None
    train_losses, train_accuracies, val_accuracies = [], [], []

    for epoch in range(EPOCHS):
        probe.train()
        correct, total, total_loss = 0, 0, 0.0
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
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}

        print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {total_loss / total:.4f} "
              f"Train accuracy: {train_acc:.4f} Validation accuracy: {val_acc:.4f}")

    probe.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val:.4f} (epoch {best_epoch})")
    plot_training(train_losses, train_accuracies, val_accuracies, out_dir)
    return probe, best_epoch

# Calculate and save accuracy and F1 scores for a given dataset
def evaluate_results(probe, X, y, dataset_name, out_dir):
    device = next(probe.parameters()).device
    probe.eval()

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    predictions = []
    true_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            outputs = probe(x_batch.to(device))
            batch_predictions = outputs.argmax(1).cpu().tolist()

            predictions.extend(batch_predictions)
            true_labels.extend(y_batch.tolist())

    accuracy = sum(
        actual == predicted
        for actual, predicted in zip(true_labels, predictions)) / len(true_labels)

    macro_f1 = f1_score(
        true_labels, predictions, average="macro", zero_division=0)

    weighted_f1 = f1_score(
        true_labels, predictions, average="weighted", zero_division=0)

    print(f"\nSCALAR MIXING - {dataset_name.upper()}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    with open(out_dir / "results.txt", "a") as f:
        f.write(f"{dataset_name.title()} Accuracy: {accuracy:.4f}\n")
        f.write(f"{dataset_name.title()} Macro F1: {macro_f1:.4f}\n")
        f.write(f"{dataset_name.title()} Weighted F1: {weighted_f1:.4f}\n")

    return accuracy, true_labels, predictions

# Generate and save the confusion matrix for the test set
def plot_confusion_matrix(y_true, y_pred, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(POS_TAGS)))
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=POS_TAGS)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    plt.title("Scalar Mixing - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()


# Load datasets, train and evaluate the POS probe, and save all results
def main():
    data_dir = BASE_DIR / "embeddings"
    out_dir = OUTPUT_DIR / "pos"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Task: POS | Source: {data_dir}")

    X_train, y_train = load_embeddings("train", data_dir)
    X_val, y_val = load_embeddings("validation", data_dir)
    X_test, y_test = load_embeddings("test", data_dir)

    probe, best_epoch = train_probe(
        X_train, y_train, X_val, y_val, len(POS_TAGS), out_dir)

    with open(out_dir / "results.txt", "w") as f:
        f.write(f"Task: pos\nBest epoch: {best_epoch}\n")

    evaluate_results(probe, X_val, y_val, "validation", out_dir)

    test_accuracy, true_labels, predictions = evaluate_results(
        probe, X_test, y_test, "test", out_dir)

    save_layer_weights(probe, out_dir)
    plot_confusion_matrix(true_labels, predictions, out_dir)

    print(f"\nResults saved to {out_dir}")


main()