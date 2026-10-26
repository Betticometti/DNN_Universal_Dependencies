import os
import matplotlib.pyplot as plt

# Read the test accuracy from a saved results file
def read_accuracy(file_path):

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Test Accuracy:"):
                return float(line.split(":")[1].strip())

    raise ValueError(f"Test Accuracy not found in {file_path}")

# Calculate and compare POS accuracy, control accuracy, and selectivity across all layers
def calculate_selectivity():

    layers = []
    pos_accuracies = []
    control_accuracies = []
    selectivities = []

    for layer in range(1, 13):

        pos_accuracy = read_accuracy(
            f"accuracy/pos_layer{layer}.txt"
        )

        control_accuracy = read_accuracy(
            f"accuracy/control_layer{layer}.txt"
        )

        selectivity = pos_accuracy - control_accuracy

        layers.append(layer)
        pos_accuracies.append(pos_accuracy)
        control_accuracies.append(control_accuracy)
        selectivities.append(selectivity)

        print(f"\nLayer {layer}")
        print(f"POS accuracy:     {pos_accuracy:.4f}")
        print(f"Control accuracy: {control_accuracy:.4f}")
        print(f"Selectivity:      {selectivity:.4f}")

    plt.figure(figsize=(12, 6))

    plt.plot(layers, pos_accuracies, marker="o",
             label="POS Accuracy")

    plt.plot(layers, control_accuracies, marker="o",
             label="Control Accuracy")

    plt.plot(layers, selectivities, marker="o",
             label="Selectivity")

    plt.xlabel("BERT Layer")
    plt.ylabel("Accuracy / Selectivity")
    plt.title("POS Accuracy, Control Accuracy and Selectivity")

    plt.xticks(layers)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/selectivity.png")
    plt.close()

    print("\nPlot saved in plots/selectivity.png")

calculate_selectivity()