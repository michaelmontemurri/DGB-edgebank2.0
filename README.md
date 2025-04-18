# EdgeBank2.0

Extending the work from "Towards Better Evaluation for Dynamic Link Prediction"

[![NeurIPS](https://img.shields.io/badge/NeurIPS-OpenReview-red)](https://openreview.net/forum?id=1GVpwr2Tfdg)
[![arXiv](https://img.shields.io/badge/arXiv-2205.12454-b31b1b.svg)](https://arxiv.org/pdf/2207.10128.pdf)
[![Video Link](https://img.shields.io/static/v1?label=Video&message=YouTube&color=red&logo=youtube)](https://www.youtube.com/watch?v=nGBP_JjKGQI)
[![Blog Post](https://img.shields.io/badge/Medium-Blog-brightgreen)](https://medium.com/@shenyanghuang1996/towards-better-link-prediction-in-dynamic-graphs-cdb8bb1e24e9)

* All dynamic graph datasets can be downloaded from [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o).

## Introduction

In this paper, the authors develop 2 versions of a simple baseline for temporal graph link prediction "EdgeBank".
The idea is to see how well pure memorization can do on a dataset. For each snapshot, predict a link if a) The link has ever existed before. b) The link has existed within a certain timewindow. 

These methods emphasize the power of pure memorization on temporal graph task, and act as simple but strong baselines, since any reasonable model should be able to outperform them. I create a simple extension that instead of taking a binary "has or hasnt existed" approach, we predict a link based on the frequency the link has existed in past snap shots. We apply the same approach to the window based method. I compare the resuslts of these extended baselines to those in the original paper.  

## Citation
```bibtex
@inproceedings{dgb_neurips_D&B_2022,
    title={Towards Better Evaluation for Dynamic Link Prediction},
    author={Poursafaei, Farimah and Huang, Shenyang and Pelrine, Kellin and and Rabbany, Reihaneh},
    booktitle={Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks},
    year={2022}
}
```
