#!/bin/bash
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH -p gpu
#SBATCH --gres=gpu:4090:2

module purge
module load ollama
module list

source ~/.bashrc
conda activate vollama

# Inicia o servidor Ollama
ollama serve &

# Espera ativa para verificar se o Ollama pronto
while ! nc -z localhost 11434; do   
  echo "Esperando o modelo iniciar..."
  sleep 5
done

echo "Modelo iniciado, pronto para execu"

# Inicia a execo do modelo
ollama run mistral-large:123b-instruct-2407-q2_K

echo "Iniciando o curl!"

curl http://127.0.0.1:11434/api/models

python -u script_extraction_llm.py