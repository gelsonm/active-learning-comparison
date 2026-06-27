# Active Learning: Uncertainty-Based Query Strategies for Label-Efficient Classification

A rigorous comparison of active learning query strategies against passive random sampling, demonstrating how smarter annotation decisions dramatically reduce labelling costs while maintaining high predictive accuracy.

## Overview

Labelling data is expensive - especially in scientific domains where expert annotation time is precious. Active learning addresses this by strategically selecting *which* unlabelled samples to annotate, prioritising the most informative ones.

This project benchmarks four strategies on a multi-class text classification task:

| Strategy | Description |
|---|---|
| **Random Sampling** | Uniform random selection from unlabelled pool (baseline) |
| **Max Entropy Sampling** | Query the sample with highest predictive entropy |
| **Margin Sampling** | Query where the top-2 class probability gap is smallest |
| **Least Confidence** | Query where the model's top class probability is lowest |

## Key Results

- Uncertainty-based strategies consistently outperform random sampling
- **Entropy sampling** reaches 85% accuracy with **~30–40% fewer labels** than random
- The uncertainty distribution plots confirm that entropy sampling targets the decision boundary - not easy, redundant examples

## Dataset

**20 Newsgroups** (4-class subset): religion, politics, sport, science. TF-IDF features (5000 dimensions).

## Running the Notebook

```bash
pip install scikit-learn numpy matplotlib seaborn
jupyter notebook active_learning_comparison.ipynb
```

## Outputs

| File | Description |
|---|---|
| `learning_curves.png` | Accuracy vs. labels used for all strategies |
| `uncertainty_distributions.png` | Distribution of entropy scores for entropy-queried vs. random-queried samples |

## Connection to Research

Active learning is the practical foundation of **Bayesian experimental design** - instead of human labels, we query expensive scientific experiments. The core principle is identical: *select the most informative next observation given current uncertainty.* This connects to probabilistic ML research on sequential experimental design and human-in-the-loop learning.
