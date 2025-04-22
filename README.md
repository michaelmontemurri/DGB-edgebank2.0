# EdgeBank2.0

Extending the work from _"Towards Better Evaluation for Dynamic Link Prediction"_

[![NeurIPS](https://img.shields.io/badge/NeurIPS-OpenReview-red)](https://openreview.net/forum?id=1GVpwr2Tfdg)
[![arXiv](https://img.shields.io/badge/arXiv-2205.12454-b31b1b.svg)](https://arxiv.org/pdf/2207.10128.pdf)
[![Video Link](https://img.shields.io/static/v1?label=Video&message=YouTube&color=red&logo=youtube)](https://www.youtube.com/watch?v=nGBP_JjKGQI)
[![Blog Post](https://img.shields.io/badge/Medium-Blog-brightgreen)](https://medium.com/@shenyanghuang1996/towards-better-link-prediction-in-dynamic-graphs-cdb8bb1e24e9)

> All dynamic graph datasets can be downloaded from [here (Zenodo)](https://zenodo.org/record/7213796#.Y1cO6y8r30o).

---

## 📘 Introduction

This project extends the original **EdgeBank** baseline from the NeurIPS Datasets & Benchmarks paper. In the original work, the authors propose two memorization-based strategies for temporal link prediction:

- **EdgeBank_inf**: A link is predicted if it has *ever* existed in the past.
- **EdgeBank_tw**: A link is predicted if it has existed *within a fixed time window*.

These approaches offer strong baselines for evaluating dynamic graph models, relying solely on link recurrence patterns.

### 🔧 Our Contributions

I introduce two **frequency-based** variants of EdgeBank that take into account *how often* a link has appeared rather than just whether it appeared:

- **`freq_weight`**: Assigns higher confidence to links that appeared more frequently over the entire history.
- **`window_freq_weight`**: Applies frequency weighting *within a moving time window*.

---

## 📊 Results

I evaluate the original and proposed EdgeBank variants on 12 diverse dynamic graph datasets. The figure below shows the **mean AUC-ROC** scores per dataset for each method:

<div align="center">
  <img src="EdgeBank/images/edgebank_aucroc_custom_grouped.png" width="80%">
</div>

As shown above, my proposed `freq_weight*` method outperforms the original `unlim_mem` (EdgeBank_inf)` on every single dataset tested (by a large margin on some datasets). Similarly, `window_freq_weight*` consistently improves upon the `time_window` (EdgeBank_tw) baseline.

To summarize overall performance across all datasets, I also compute the average AUC-ROC (± std deviation) per memory strategy:

<div align="center">
  <img src="EdgeBank/images/edgebank_aucroc_by_strategy.png" width="60%">
</div>

> **Note**: Asterisk (*) denotes my new frequency-based additions.

---

## 📌 Citation

```bibtex
@inproceedings{dgb_neurips_D&B_2022,
    title={Towards Better Evaluation for Dynamic Link Prediction},
    author={Poursafaei, Farimah and Huang, Shenyang and Pelrine, Kellin and and Rabbany, Reihaneh},
    booktitle={Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks},
    year={2022}
}

