import os
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Configurazione
# ==========================================
SPLIT = "train"
SENTENCE = 0  
BASE_DIR = "../target_layers"

# Definisci quali layer hai salvato e vuoi confrontare
# (Se hai salvato solo i layer 1, 2, 3 e 4, metti quelli)
AVAILABLE_LAYERS = [1, 2, 3, 4] 

# ==========================================
# 2. Funzione per calcolare la statistica (Norma media)
# ==========================================
def calculate_layer_statistic(layer_num, sentence_id=None):
    """
    Carica i dati del layer e calcola una metrica numerica (es. la media della 
    norma (grandezza) dei vettori) per poterla plottare in un grafico.
    """
    layer_dir = os.path.join(BASE_DIR, f"layer_{layer_num}")
    
    if sentence_id is not None:
        # Cerca la frase specifica
        search_pattern = os.path.join(layer_dir, f"{SPLIT}_sentence_{sentence_id}.pt")
    else:
        # Cerca tutte le frasi
        search_pattern = os.path.join(layer_dir, f"{SPLIT}_sentence_*.pt")
        
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        print(f"Attenzione: Nessun file trovato per il layer {layer_num}")
        return None
        
    norms = []
    for file_path in file_paths:
        try:
            data = torch.load(file_path, weights_only=False)
            hidden_states = data['hidden_states'] # Shape: (1, num_tokens, 768)
            
            # Togliamo la dimensione del batch: (num_tokens, 768)
            hidden_states = hidden_states.squeeze(0)
            
            # Calcoliamo la norma (intensità) del vettore per ogni token
            # dim=1 significa "calcola lungo la dimensione dei 768 numeri"
            token_norms = torch.norm(hidden_states, p=2, dim=1) 
            
            # Calcoliamo la media delle norme per questa intera frase
            avg_sentence_norm = torch.mean(token_norms).item()
            norms.append(avg_sentence_norm)
            
        except Exception as e:
            print(f"Errore nel caricamento di {file_path}: {e}")
            
    # Se stiamo processando più frasi, facciamo la media finale di tutte le frasi
    return np.mean(norms) if norms else None

# ==========================================
# 3. Estrazione dati
# ==========================================
print(f"--- Avvio Analisi per {f'Frase {SENTENCE}' if SENTENCE is not None else 'Tutte le Frasi'} ---")

layer_values = []
valid_layers = []

for layer in AVAILABLE_LAYERS:
    print(f"Elaborazione Layer {layer}...")
    val = calculate_layer_statistic(layer, SENTENCE)
    if val is not None:
        layer_values.append(val)
        valid_layers.append(layer)

# ==========================================
# 4. Creazione del Plot
# ==========================================
if valid_layers:
    plt.figure(figsize=(10, 6))
    
    # Crea un grafico a linee con dei punti (marker='o')
    plt.plot(valid_layers, layer_values, marker='o', linestyle='-', color='b', linewidth=2)
    
    title = f"Confronto Intensità degli Embeddings per i Layer BERT"
    if SENTENCE is not None:
        title += f"\n(Split: {SPLIT} - Frase: {SENTENCE})"
    else:
        title += f"\n(Media su tutte le frasi - Split: {SPLIT})"
        
    plt.title(title, fontsize=14)
    plt.xlabel("Indice del Layer", fontsize=12)
    plt.ylabel("Norma L2 Media (Intensità Vettore)", fontsize=12)
    
    # Imposta i "tick" dell'asse X esattamente sui layer che hai analizzato
    plt.xticks(valid_layers)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Salva l'immagine invece di provare a mostrarla a schermo (fondamentale su server)
    plot_path = f"confronto_layer_{'frase_'+str(SENTENCE) if SENTENCE is not None else 'tutte'}.png"
    plt.savefig(plot_path)
    print(f"\nGrafico salvato con successo come: {plot_path}")
    
else:
    print("Nessun dato valido trovato per generare il grafico.")