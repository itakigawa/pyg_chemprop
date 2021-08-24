# ChemProp in PyTorch Geometric

A concise and easy-to-customize reimplementation of "ChemProp" (Yang et al, 2019) in [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).

# Features

- "pyg_chemprop_utils" includes a converter from smiles to a pyg object defining a molecular graph with atom- and bond- features in the original ChemProp. (this requires [RDKit](https://github.com/rdkit/rdkit))
- "pyg_chemprop.py" uses [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter), and requires the "index of reverse edges" for ChemProp. You'll need to preprocess pyg datasets (or lists of pyg objects) like

```python
from pyg_chemprop import RevIndexedDataset
from ogb.graphproppred import PygGraphPropPredDataset
pyg_dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root="dataset/")
dataset = RevIndexedDataset(pyg_dataset)
```
- "pyg_chemprop_naive.py" does not use [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter). It's very slow, but easy to understand what is going on inside ChemProp.

# Usage

See "test_ogbg-molhiv.ipynb".

# Speed Test

Environment

* torch 1.8.1
* torch_geometric 1.7.0
* torch_scatter 2.0.7
* ogb 1.3.1

Results

- data: "ogbg-molhiv" training dataset (32,901 molecules)
- batch_size: 50
- gpu: A100-PCIE-40GB

|  | device | features | time per epoch |
|:-----------|------------:|------------:|:------------:|
| original chemprop  | CPU        | chemprop | 70 sec         |
|            | GPU        | chemprop | 15 sec       |
| ours (w pytorch_scatter)       | CPU        | ogb default | **28 sec**         |
|          | GPU          | ogb default | **5 sec**           |
| ours (w pytorch_scatter)       | CPU        | chemprop |  **59 sec**         |
|          | GPU          | chemprop | **7 sec**           |
| ours (w/o pytorch_scatter)     | CPU       | ogb default | 1277 sec       |
|     | GPU     | ogb default | 1743 sec      |


# ChemProp (Yang et al, 2019)

"ChemProp" is a simple but effective Graph Neural Network (GNN) for Molecular Property Prediction, and was successfully used in anti-biotic discovery by Machine Learning for Pharmaceutical Discovery and Synthesis Consortium (MLPDS), MIT.

- The original code: https://github.com/chemprop/chemprop
- Yang et al (2019). Analyzing Learned Molecular Representations for Property Prediction. *JCIM*, 59(8), 3370–3388. https://doi.org/10.1021/acs.jcim.9b00237
- Yang et al (2019). Correction to Analyzing Learned Molecular Representations for Property Prediction. *JCIM*, 59(12), 5304–5305. https://doi.org/10.1021/acs.jcim.9b01076
- Stokes et al (2020). A Deep Learning Approach to Antibiotic Discovery. *Cell*, 180(4), 688–702.e13. https://doi.org/10.1016/j.cell.2020.01.021
- Marchant (2020). Powerful antibiotics discovered using AI. *Nature*, https://doi.org/10.1038/d41586-020-00018-3

# Author

* Ichigaku Takigawa

