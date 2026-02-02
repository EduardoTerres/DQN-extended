# Continuous Query Decomposition [Re]

---

## Introduction

This repository contains a reproduction and extension of the [Complex Query Answering with Neural Link Predictors (CQD)](https://openreview.net/forum?id=Mos9F9kDwkz) paper. 

The project includes:
- **Reproduction code** in `src/reproduction/`
- **Density modeling extensions** in `src/density/`
- **Additional extensions** in `src/extension/`
- **Knowledge base completion code** in `src/kbc/` - adapted from the [original kbc repository](https://github.com/facebookresearch/kbc) with modifications

## Preliminaries
### 1. Install the requirements

We recommend creating a new environment:

```bash
% conda create --name cqd python=3.8 && conda activate cqd
% pip install -r requirements.txt
```

### 2. Download the data

We use 3 knowledge graphs: FB15k, FB15k-237, and NELL.
From the root of the repository, download and extract the files to obtain the folder `data`, containing the sets of triples and queries for each graph.

```bash
% wget http://data.neuralnoise.com/cqd-data.tgz
% tar xvf cqd-data.tgz
```

### 3. Download the models

Then you need neural link prediction models -- one for each of the datasets.
Our pre-trained neural link prediction models are available here:

```bash
% wget http://data.neuralnoise.com/cqd-models.tgz
% tar xvf cqd-models.tgz
```

## 2. Train the density estimation model (flow model)
First export the src folder to path
```
export PYTHONPATH=src/
```

To train the model, with logging to wandb use:

```
python src/density/flow.py
```

To obtain statistics on the evolution of the log-likelihood of the entity embeddings onthe trained flow matching model:
```
python src/density/likelihood.py
```
## 2. Answering queries with the new density regularization
To perform batch scoring of the selected datasets use 

```
python src/extension/train.py
```

Finally, obtain the results and statistics on them using:
```
python src/extension/results.py
```

Here is the citation for the original work:
```bibtex
@inproceedings{
    arakelyan2021complex,
    title={Complex Query Answering with Neural Link Predictors},
    author={Erik Arakelyan and Daniel Daza and Pasquale Minervini and Michael Cochez},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=Mos9F9kDwkz}
}
```
