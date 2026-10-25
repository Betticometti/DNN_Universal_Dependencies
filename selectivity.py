
import torch

LAYER_NUMBER = 2

# Calculate selectivity
def calculate_selectivity():

    pos_data = torch.load(
        f"accuracy/pos_layer{LAYER_NUMBER}.pt",
        weights_only=True
    )

    control_data = torch.load(
        f"accuracy/control_layer{LAYER_NUMBER}.pt",
        weights_only=True
    )

    pos_accuracy = pos_data["validation_accuracy"]
    control_accuracy = control_data["validation_accuracy"]

    selectivity = pos_accuracy - control_accuracy

    print(f"Layer {LAYER_NUMBER}")
    print(f"POS accuracy:     {pos_accuracy:.4f}")
    print(f"Control accuracy: {control_accuracy:.4f}")
    print(f"Selectivity:      {selectivity:.4f}")


calculate_selectivity()
