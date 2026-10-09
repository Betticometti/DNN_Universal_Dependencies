import os
# device number 3 for group 7-8 (we are 8)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("\n\n------------ OUTPUT ------------\n\n")

# START OF CODE
name = "Riccardo"

print("Hello, " + name + "!")
print("Python is running correctly.")
print("2 + 3 =", 2 + 3)

for i in range(5):
    print("Iteration", i + 1)