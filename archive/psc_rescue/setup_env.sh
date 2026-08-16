#!/bin/bash

echo "Loading Anaconda and activating virtual environment..."
module load anaconda3
conda activate /ocean/projects/cis260125p/shared/envs/project

echo "Setting Python and Shared environment variables..."
export PYTHONNOUSERSITE=1
export SHARED=/ocean/projects/cis260125p/shared

echo "Setting up Ollama paths..."
export PATH=$SHARED/ollama/bin:$PATH
export LD_LIBRARY_PATH=$SHARED/ollama/lib:$LD_LIBRARY_PATH
export OLLAMA_MODELS=$SHARED/ollama_models

echo "Navigating to project directory..."
cd $SHARED/assessment-tool-speech-dataset/

echo "✅ Setup complete! You can now run 'ollama serve &'."