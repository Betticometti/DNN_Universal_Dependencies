# ==========================================
# Utilities
# ==========================================

import os
import torch

# ==========================================
# Gorina Environment
# ==========================================

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# Loading Layer Data
# ==========================================

TARGET_LAYER = 2
SPLIT = "train"
SENTENCE = 0

file_path = f"./target_layers/layer_{TARGET_LAYER}/{SPLIT}_sentence_{SENTENCE}.pt"
data = torch.load(file_path, weights_only=False)

print(f"\n --- OPENING FILE: {file_path} --- \n")
print("Chiavi presenti:", data.keys())
print("\n1. POS tags della frase (Parole originali):")
print(data['pos_tags'])
print(f"-> Totale parole: {len(data['pos_tags'])}")

print("\n2. Word IDs (Mappatura dei sub-token di BERT):")
print(data['word_ids'])
print(f"-> Totale sub-token generati: {len(data['word_ids'])}")

print("\n3. Dimensioni del tensore BERT (Hidden States):")
print(data['hidden_states'].shape)
print("-> (batch_size, numero_di_sub_token, dimensione_embedding)")