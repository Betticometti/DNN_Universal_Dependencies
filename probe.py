import os

from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

LAYER_NUMBER = 1
CONTROL_TASK = False
PROBE_TYPE = "mlp" # "mlp" "linear"

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


# Probe for POS classification (linear or MLP)
class Probe(nn.Module):
    # hidden_dim was 128 before, but hewitt and liang talked about having it way lower
    def __init__(self, input_dim, num_classes, probe_type="linear", hidden_dim=10):
        super().__init__()
        self.probe_type = probe_type
        
        if self.probe_type == "linear":
            self.model = nn.Linear(input_dim, num_classes)
        elif self.probe_type == "mlp":
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            raise ValueError("probe_type must be 'linear' or 'mlp'")

    def forward(self, x):
        return self.model(x)

# Calculate classification accuracy on a given dataset
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

# Save the best validation accuracy to a text file
def save_accuracy(accuracy):
    task_name = "control" if CONTROL_TASK else "pos"
    folder_path = f"accuracy/{PROBE_TYPE}/{task_name}"
    
    # Crea l'intero percorso fino a "control" o "pos" in automatico
    os.makedirs(folder_path, exist_ok=True)

    file_name = f"{folder_path}/{PROBE_TYPE}_{task_name}_layer{LAYER_NUMBER}.txt"

    with open(file_name, "w") as f:
        f.write(f"Validation Accuracy: {accuracy:.4f}\n")

    print(f"Accuracy saved: {file_name}")

# Train the linear probe and restore the model with the best validation accuracy
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

    probe = Probe(X_train.shape[1], num_classes, probe_type=PROBE_TYPE).to(device)

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

    # Restore the best probe based on validation accuracy
    probe.load_state_dict(best_state)

    print(f"Best validation accuracy: {best_val:.4f}")

    plot_training(train_losses, train_accuracies, val_accuracies)
    save_accuracy(best_val)

    return probe

# Plot and save training loss, training accuracy, and validation accuracy
def plot_training(train_losses, train_accuracies, val_accuracies):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss ({PROBE_TYPE})")
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training accuracy")
    plt.plot(epochs, val_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracy ({PROBE_TYPE})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    task_name = "control" if CONTROL_TASK else "pos"
    plot_dir = f"plots/{PROBE_TYPE}/{task_name}"

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/probe_{PROBE_TYPE}_layer{LAYER_NUMBER}_training.png")
    plt.close()

# Evaluate accuracy and F1 scores and display classification errors
def show_errors(probe, X, y, id_to_pos, dataset_name, max_errors=30):

    device = next(probe.parameters()).device
    probe.eval()

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    errors = []
    predictions = []
    true_labels = []

    index = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:

            outputs = probe(x_batch.to(device))
            batch_predictions = outputs.argmax(1).cpu().tolist()
            batch_labels = y_batch.tolist()

            predictions.extend(batch_predictions)
            true_labels.extend(batch_labels)

            for true_label, prediction in zip(batch_labels, batch_predictions):
                if prediction != true_label:
                    errors.append((index, true_label, prediction))

                index += 1

    # Evaluation metrics
    accuracy = 1 - len(errors) / len(y)
    macro_f1 = f1_score(
        true_labels, predictions,
        average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        true_labels, predictions,
        average="weighted", zero_division=0
    )

    # Save test results
    if dataset_name == "test":

        task_name = "control" if CONTROL_TASK else "pos"
        os.makedirs(f"accuracy/{PROBE_TYPE}/{task_name}", exist_ok=True)
        file_name = f"accuracy/{PROBE_TYPE}/{task_name}/{PROBE_TYPE}_{task_name}_layer{LAYER_NUMBER}.txt"

        with open(file_name, "a") as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Test Macro F1: {macro_f1:.4f}\n")
            f.write(f"Test Weighted F1: {weighted_f1:.4f}\n")

    print("\n" + "=" * 60)
    print(f"PROBE EVALUATION ({PROBE_TYPE.upper()}) - {dataset_name.upper()} SET")
    print("=" * 60)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\n" + "=" * 60)
    print(f"CLASSIFICATION ERRORS - {dataset_name.upper()} SET")
    print("=" * 60)

    print(f"Total errors: {len(errors)}")
    print(f"Total words: {len(y)}")

    print(f"\nFirst {min(max_errors, len(errors))} errors:")
    print(f"{'Index':<10}{'True label':<15}{'Predicted label':<15}")
    print("-" * 40)

    for i, true_label, prediction in errors[:max_errors]:
        print(
            f"{i:<10}"
            f"{id_to_pos[true_label]:<15}"
            f"{id_to_pos[prediction]:<15}"
        )

# Generate and save the confusion matrix for the test set
def plot_confusion_matrix(probe, X_test, y_test, pos_tags):

    device = next(probe.parameters()).device
    probe.eval()

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    predictions = []
    true_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            outputs = probe(x)
            predicted = outputs.argmax(1)

            predictions.extend(predicted.cpu().tolist())
            true_labels.extend(y.tolist())

    # Compute confusion matrix
    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=range(len(pos_tags))
    )

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=pos_tags
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d"
    )

    plt.title(f"Confusion Matrix - Layer {LAYER_NUMBER} ({PROBE_TYPE}) Test Set")
    plt.tight_layout()

    task_name = "control" if CONTROL_TASK else "pos"
    plot_dir = f"plots/{PROBE_TYPE}/{task_name}"

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/confusion_matrix_{PROBE_TYPE}_layer{LAYER_NUMBER}.png")
    plt.close()

POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET",
    "INTJ", "NOUN", "NUM", "PART", "PRON",
    "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]

id_to_pos = {i: tag for i, tag in enumerate(POS_TAGS)}

# single run of training and evaluation for a specific layer and task
""" # Load embeddings
if CONTROL_TASK:
    data_dir = "control_embeddings"
else:
    data_dir = "embeddings"

train_data = torch.load(
    f"{data_dir}/train_layer{LAYER_NUMBER}.pt",
    weights_only=True
)

val_data = torch.load(
    f"{data_dir}/validation_layer{LAYER_NUMBER}.pt",
    weights_only=True
)

test_data = torch.load(
    f"{data_dir}/test_layer{LAYER_NUMBER}.pt",
    weights_only=True
)

X_train = train_data["embeddings"]
y_train = train_data["labels"]

X_val = val_data["embeddings"]
y_val = val_data["labels"]

X_test = test_data["embeddings"]
y_test = test_data["labels"]


# Train the probe
probe = train_probe(X_train, y_train, X_val, y_val, 17)

# Analyze classification errors on validation set
show_errors(probe, X_val, y_val, id_to_pos, "validation")

# Analyze classification errors on test set
show_errors(probe, X_test, y_test, id_to_pos, "test")

# Plot confusion matrix on test set
plot_confusion_matrix(probe, X_test, y_test, POS_TAGS) """

# Train and evaluate POS and control probes across all BERT layers
def run_all_training_eval_tests():
    global LAYER_NUMBER, CONTROL_TASK

    for i in range(1, 13):
        if not os.path.exists(f"embeddings/train_layer{i}.pt") \
            or not os.path.exists(f"embeddings/validation_layer{i}.pt") \
            or not os.path.exists(f"embeddings/test_layer{i}.pt") \
            or not os.path.exists(f"control_embeddings/train_layer{i}.pt") \
            or not os.path.exists(f"control_embeddings/validation_layer{i}.pt") \
            or not os.path.exists(f"control_embeddings/test_layer{i}.pt") \
            :
            print(f"Missing embeddings for layer {i} - please run extract_embeddings.py and control_task.py first.")
            exit(1)

        LAYER_NUMBER = i

        for CONTROL_TASK in [False, True]:

            # Load embeddings
            if CONTROL_TASK:
                data_dir = "control_embeddings"
                print(f"\n\nRunning probe for control task - layer {LAYER_NUMBER}")
            else:
                print(f"\n\nRunning probe for POS classification - layer {LAYER_NUMBER}")
                data_dir = "embeddings"

            train_data = torch.load(
                f"{data_dir}/train_layer{LAYER_NUMBER}.pt",
                weights_only=True
            )

            val_data = torch.load(
                f"{data_dir}/validation_layer{LAYER_NUMBER}.pt",
                weights_only=True
            )

            test_data = torch.load(
                f"{data_dir}/test_layer{LAYER_NUMBER}.pt",
                weights_only=True
            )

            X_train = train_data["embeddings"]
            y_train = train_data["labels"]

            X_val = val_data["embeddings"]
            y_val = val_data["labels"]

            X_test = test_data["embeddings"]
            y_test = test_data["labels"]

            # Train the probe
            probe = train_probe(X_train, y_train, X_val, y_val, 17)

            # Analyze classification errors on validation set
            show_errors(probe, X_val, y_val, id_to_pos, "validation")

            # Analyze classification errors on test set
            show_errors(probe, X_test, y_test, id_to_pos, "test")

            # Plot confusion matrix on test set
            plot_confusion_matrix(probe, X_test, y_test, POS_TAGS)

    return

run_all_training_eval_tests()