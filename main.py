# ==========================================
# Gorina Environment
# ==========================================

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# Installations
# ==========================================

pip install -U transformer
pip install conllu

# ==========================================
# Utilities
# ==========================================

import os, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

# ==========================================
# Loading Dataset
# ==========================================
print("Loading dataset split from folder 'data'...")

data_files = {
    "train": "data/en_ewt-ud-train.conllu",
    "validation": "data/en_ewt-ud-dev.conllu", 
    "test": "data/en_ewt-ud-test.conllu"
}

# script 'conllu' from Hugging Face to read dataset files
dataset = load_dataset("conllu", data_files=data_files)

# Verifica rapida che stia leggendo correttamente
print(dataset['train'][0]['tokens'])

# ==========================================
# Model Configuration
# ==========================================

MODEL_NAME = "bert-base-uncased"

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
TARGET_LAYER = 8  # Scegli quale layer studiare (da 0 a 12 per BERT base)
SAVE_DIR = f"./hidden_states/layer_{TARGET_LAYER}"

# Creiamo la cartella se non esiste
os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# Model Loader
# ==========================================

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True) # 
model.to(device) # directly on gpu
model.eval() # set model to eval mode (disable dropout etc.)


