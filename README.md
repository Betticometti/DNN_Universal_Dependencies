# DNN_Universal_Dependencies
**ELE680 Deep Neural Networks – Probing Language Model**

**Group 8**

## Dataset and Model

- **Dataset:** Universal Dependencies English Web Treebank (UD English-EWT)
- **Model:** `google-bert/bert-base-uncased`
- **Task:** Part-of-Speech classification (17 Universal POS tags)
- **Classifier:** Linear probe and Multi-Layer Perceptron (MLP)

## Main Files

| File | Description |

| `test_bert.py` | Loads the dataset, tokenizer, and pretrained BERT model. |
| `extract_embeddings.py` | Extracts and saves word embeddings from BERT layers. |
| `control_task.py` | Creates random POS labels for the control task. |
| `probe.py` | Trains and evaluates POS and control probes. |
| `selectivity.py` | Calculates selectivity and generates comparison plots. |
| `EDA/eda_dataset.py` | Analyzes dataset sizes and POS tag distributions. |
| `EDA/eda_tokenization.py` | Analyzes WordPiece tokenization and sentence lengths. |

## How to Run

Run the following commands from the project root directory, in this order:

`python extract_embeddings.py`
Extracts word embeddings and POS labels from the selected BERT layers and saves them in the `embeddings/` directory.

`python control_task.py`
Assigns random but consistent labels to words and saves the resulting datasets in `control_embeddings/`.

In probe.py, select the architecture:
PROBE_TYPE = "linear" or "mlp"
Then run: `python probe.py`
Trains and evaluates both POS and control probes for the selected architecture.
Results are saved in:
`accuracy/linear/pos/` and `accuracy/linear/control/`
`accuracy/mlp/pos/` and `accuracy/mlp/control/`
`plots/linear/` and `plots/mlp/`
Evaluation includes validation accuracy, test accuracy, Macro F1, Weighted F1, training curves, and confusion matrices.

In selectivity.py, select the architecture:
PROBE_TYPE = "linear" or "mlp"
Then run: `python selectivity.py`
Calculates selectivity as the difference between POS test accuracy and control test accuracy for each layer.
The comparison plots are saved in:
`plots/linear/selectivity/` and in `plots/mlp/selectivity/`