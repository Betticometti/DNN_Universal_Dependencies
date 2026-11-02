import os
import matplotlib.pyplot as plt

PROBE_TYPE = "mlp"   # "linear" oppure "mlp"

# Read the test accuracy from a saved results file
def read_accuracy(file_path):
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Test Accuracy:"):
                return float(line.split(":")[1].strip())

    raise ValueError(f"Test Accuracy not found in {file_path}")


# Read evaluation metrics from a saved results file
def read_metrics(file_path):
    metrics = {}

    with open(file_path, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                metrics[key] = float(value.strip())

    return (
        metrics["Test Accuracy"],
        metrics["Test Macro F1"],
        metrics["Test Weighted F1"]
    )


# Plot POS performance across all BERT layers
def plot_pos_performance():

    layers = []
    accuracies = []
    macro_f1 = []
    weighted_f1 = []

    for layer in range(1, 13):

        file_path = (
            f"accuracy/{PROBE_TYPE}/pos/"
            f"{PROBE_TYPE}_pos_layer{layer}.txt"
        )

        accuracy, macro, weighted = read_metrics(file_path)

        layers.append(layer)
        accuracies.append(accuracy)
        macro_f1.append(macro)
        weighted_f1.append(weighted)

    plt.figure(figsize=(12, 6))

    plt.plot(layers, accuracies, marker="o", linestyle="--",
         label="Test Accuracy", zorder=3)

    plt.plot(layers, macro_f1, marker="s", linestyle="-",
         label="Test Macro F1")

    plt.plot(layers, weighted_f1, marker="^", linestyle=":",
         label="Test Weighted F1")

    plt.xlabel("BERT Layer")
    plt.ylabel("Score")
    plt.title(f"POS Tagging Performance ({PROBE_TYPE.upper()} Probe)")

    plt.xticks(layers)
    plt.ylim(0.85, 1.0)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plot_dir = f"accuracy/{PROBE_TYPE}/pos"
    os.makedirs(plot_dir, exist_ok=True)

    plot_path = f"{plot_dir}/{PROBE_TYPE}_pos_performance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nPlot saved in {plot_path}")


# Calculate and compare POS accuracy, control accuracy, and selectivity across all layers
def calculate_selectivity():

    layers = []
    pos_accuracies = []
    control_accuracies = []
    selectivities = []

    for layer in range(1, 13):

        pos_file = (
            f"accuracy/{PROBE_TYPE}/pos/"
            f"pos_layer{layer}.txt"
        )

        control_file = (
            f"accuracy/{PROBE_TYPE}/control/"
            f"control_layer{layer}.txt"
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

plot_pos_performance()
calculate_selectivity()