# Lab 4: GNN
## Introduction
This project is the experimental code for Lab4, which replicates three node embedding methods: DeepWalk, LINE, 
and Node2vec, and tests the Node Classification task and Link Predication task (using the dataset and 
evaluation code from the OGB library), and simply visualizes the graph network
## Usage
### Example
if you'd like to test Node Classification task using DeepWalk, run this:
```
python src/train_embedding.py --dataset ogbn-arxiv --model deepwalk
```
then the embedding file will be saved in `saved_model/$dataset name$` directory

after that, run:
```
python src/evaluate.py --dataset obgn-arxiv --model deepwalk --runs 5 --epochs 200
```
The result will be output in the console:
```
Highest Train: 79.66
Highest Valid: 71.47
  Final Train: 78.27
   Final Test: 70.40
All runs:
Highest Train: 79.67 ± 0.07
Highest Valid: 71.43 ± 0.12
  Final Train: 78.11 ± 0.46
   Final Test: 70.25 ± 0.17
```
