# A* Heuristic generating with transformer 

In this repository, we release models from the papers

- [Learning-accelerated A* Search for Risk-aware Path Planning], which will be present in SCITECH 2024


## Introduction
This study aims to enhance the efficiency of solving the Constrained Shortest Path (CSP) problem. We introduce a novel approach that employs a transformer-based neural network, specifically a decoder architecture, to generate heuristic values for the A* algorithm. This methodological innovation results in a significant acceleration of the A* algorithm's performance, outpacing the traditional Manhattan heuristic in terms of speed.
<p align="center">
<img src="https://github.com/Xiaoshan-jun/riskmap/blob/main/figure/randommapexample8.png" width="300" height="200">
<img src="https://github.com/Xiaoshan-jun/riskmap/blob/main/figure/randommapexampleManhattan8.png" width="300" height="200">
<img src="https://github.com/Xiaoshan-jun/riskmap/blob/main/figure/randommapexampleleanred8.png" width="300" height="200">
</p>

## Generator architecture
![Alt text](https://github.com/Xiaoshan-jun/riskmap/blob/main/figure/input.PNG)
![Alt text](https://github.com/Xiaoshan-jun/riskmap/blob/main/figure/good%20figure/input.PNG)


train
```
python train.py
```
test predict, you need to change the args.model_path into the path you saved the model(end with .pt) first
```
python visualization.py 
```
formal evaluation, you need to change the arg.model_path into the path you saved the model(end with .pt) first
```
python evaluate_model.py
```




## Bibtex

```
@inproceedings{xiang2024learning,
  title={Learning-accelerated A* Search for Risk-aware Path Planning},
  author={Xiang, Jun and Xie, Junfei and Chen, Jun},
  booktitle={AIAA SCITECH 2024 Forum},
  pages={2895},
  year={2024}
}

```





