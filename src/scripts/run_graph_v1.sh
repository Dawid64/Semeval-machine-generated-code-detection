#!/bin/sh
#SBATCH --job-name=GNN-SemEval
#SBATCH --partition=hgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --comment="GNN-Semeval for Semantic Webs and Social Networks"

echo "Running model $1"
uv run semeval train $1
