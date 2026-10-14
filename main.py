# ==========================================
# Utilities
# ==========================================

import os, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from datasets import load_dataset
from conllu import parse_incr # parser for conllu files

# ==========================================
# Gorina Environment
# ==========================================

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# Installations
# ==========================================

'''
pip install -U transformers
pip install conllu
pip install scikit-learn

'''

# ==========================================
# Loading Dataset
# ==========================================

def load_data(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(parse_incr(file))

data_files = {
    "train": "data/en_ewt-ud-train.conllu",
    "validation": "data/en_ewt-ud-dev.conllu",
    "test": "data/en_ewt-ud-test.conllu"
}

print("Loading dataset split from folder 'data'...")

dataset = {
    split: load_data(path)
    for split, path in data_files.items()
}

""" print([token["form"] for token in dataset["train"][0]
       if isinstance(token["id"], int)]) """

# print all tokens of the raw sentence (usually called 'text' or 'tokens' or 'form')
print(dataset['train'][0])

# ==========================================
# Model Configuration
# ==========================================

MODEL_NAME = "bert-base-uncased"

# ==========================================
# Model Loader
# ==========================================

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True) # 
model.to(device) # directly on gpu
model.eval() # set model to eval mode (disable dropout...)

# ==========================================
# Extraction Function
# ==========================================

TARGET_LAYER = 2  # 0 to 12 for BERT base
SAVE_DIR = f"./target_layers/layer_{TARGET_LAYER}"

os.makedirs(SAVE_DIR, exist_ok=True)

def extract_and_save_layer(dataset_split, split_name, max_samples=None):
    print(f"\nBeginning extraction for split: {split_name}")
    
    with torch.no_grad(): # no gradient calculation
        for i, example in enumerate(dataset_split):
            if max_samples and i >= max_samples:
                break
            print(f"Processing sentence {i} in split {split_name}")
            print(f"example: {example}")
            # the keys are organized in a TokenList so we use form and upos
            tokens = [token['form'] for token in example]
            pos_tags = [token['upos'] for token in example]
            print(f"tokens: {tokens}")
            print(f"pos_tags: {pos_tags}")
            # using is_split_into_words=True to keep as many words together as possible
            inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
            # getting the word_ids before it is changed in py dict
            word_ids = inputs.word_ids()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            print(f"inputs: {inputs}")
            # Passaggio nel modello
            outputs = model(**inputs)
            print(f"outputs: {outputs}")
            
            # outputs.hidden_states è una tupla di 13 tensori:
            # [0] = embeddings iniziali, [1...12] = output dei 12 layer del transformer
            all_hidden_states = outputs.hidden_states
            print(f"all_hidden_states: {all_hidden_states}")
            
            # Non salvare mai i tensori sulla GPU, o finirai la memoria di Gorina4
            target_layer_tensor = all_hidden_states[TARGET_LAYER].cpu()
            print(f"target_layer_tensor: {target_layer_tensor}")
            
            # saving the extracted layer, POS tags, and word IDs for later use
            data_to_save = {
                'hidden_states': target_layer_tensor,
                'pos_tags': pos_tags,      # Universal Part-of-Speech tags
                'word_ids': word_ids     # to realign BERT sub-tokens with the original words
            }
            
            # Salvataggio su disco
            file_name = f"{split_name}_sentence_{i}.pt"
            file_path = os.path.join(SAVE_DIR, file_name)
            print(f"data_to_save: {data_to_save}")
            torch.save(data_to_save, file_path)
            
            
            print(f"Elaborata frase {i} - Salvata in: {file_path}")

# ==========================================
# Run
# ==========================================

print("\n----------- Start of Execution -----------\n")
# TEST
extract_and_save_layer(dataset['train'], 'train', max_samples=1)
# extract_and_save_layer(dataset['validation'], 'dev', max_samples=5)
# extract_and_save_layer(dataset['test'], 'test', max_samples=5)

