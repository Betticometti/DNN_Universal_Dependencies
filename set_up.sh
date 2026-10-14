echo "=== Connect to Gorina ==="
ssh gorina4

echo "=== Folder Path ==="
cd ~/bhome/DNN_Universal_Dependencies

echo "=== Conda Environment ==="
uenv miniconda3-py312
conda activate test_env_pytorch

echo "=== Python Version ==="
which python
python --version

echo "=== Git status ==="
git status

# echo "=== GPU ==="
# nvidia-smi

echo "=== Ready ==="
exec bash -i
