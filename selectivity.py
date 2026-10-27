import os
import matplotlib.pyplot as plt

PROBE_TYPE = "linear"   # "linear" oppure "mlp"

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

        pos_file = (
            f"accuracy/{PROBE_TYPE}/pos/"
            f"{PROBE_TYPE}_pos_layer{layer}.txt"
        )

        control_file = (
            f"accuracy/{PROBE_TYPE}/control/"
            f"{PROBE_TYPE}_control_layer{layer}.txt"
        )

        pos_accuracy = read_accuracy(pos_file)
        control_accuracy = read_accuracy(control_file)

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

    plt.plot(layers, pos_accuracies, marker="o", label="POS Accuracy")
    plt.plot(layers, control_accuracies, marker="o", label="Control Accuracy")
    plt.plot(layers, selectivities, marker="o", label="Selectivity")

    plt.xlabel("BERT Layer")
    plt.ylabel("Accuracy / Selectivity")
    plt.title(f"POS Accuracy, Control Accuracy and Selectivity ({PROBE_TYPE})")

    plt.xticks(layers)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plot_dir = f"plots/{PROBE_TYPE}/selectivity"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/selectivity.png")
    plt.close()

    print(f"\nPlot saved in {plot_dir}/selectivity.png")

calculate_selectivity()