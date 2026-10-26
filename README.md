# DNN_Universal_Dependencies
**ELE680 Deep Neural Networks – Probing Language Model**

**Group 8**

## Dataset and Model

- **Dataset:** Universal Dependencies English Web Treebank (UD English-EWT)
- **Model:** `google-bert/bert-base-uncased`
- **Task:** Part-of-Speech classification (17 Universal POS tags)
- **Classifier:** Linear probe

## Main Files

| File | Description |

| `test_bert.py` | Loads the dataset, tokenizer, and pretrained BERT model. |
| `extract_embeddings.py` | Extracts and saves word embeddings from BERT layers. |
| `control_task.py` | Creates random POS labels for the control task. |
| `probe.py` | Trains and evaluates POS and control probes. |
| `selectivity.py` | Calculates selectivity and generates comparison plots. |

## How to Run

Run the following commands from the project root directory, in this order:

`python extract_embeddings.py`
Extracts word embeddings and POS labels from the selected BERT layers and saves them in the `embeddings/` directory.

`python control_task.py`
Assigns random but consistent labels to words and saves the resulting datasets in `control_embeddings/`.

`python probe.py`
Trains and evaluates the POS and control probes, saving accuracy results and plots in the `accuracy/` and `plots/` directories.

`python selectivity.py`
Calculates selectivity for each layer and saves a comparison plot in `plots/selectivity.png`.