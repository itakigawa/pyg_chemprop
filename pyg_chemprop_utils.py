from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.data import Data
from tqdm import tqdm


class FeatureScaler:
    def __init__(self, targets):
        self.targets = targets
        self.m = {}
        self.u = {}

    def fit(self, dataset):
        n = {t: 0 for t in self.targets}
        s = {t: 0.0 for t in self.targets}
        ss = {t: 0.0 for t in self.targets}
        for t in self.targets:
            for data in dataset:
                n[t] += data[t].shape[0]
                s[t] += data[t].sum(dim=0)
                ss[t] += (data[t] ** 2).sum(dim=0)
            m = s[t] / n[t]
            v = (ss[t] / (n[t] - 1) - (n[t] / (n[t] - 1)) * (m ** 2)).sqrt()
            u = 1.0 / v
            u[u == float("inf")] = 1.0
            self.m[t] = m
            self.u[t] = u

    def transform(self, dataset):
        data_list = []
        for data in tqdm(dataset):
            for t in self.targets:
                data[t] = self.u[t] * (data[t] - self.m[t])
            data_list.append(data)
        return data_list

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)


def mol2data(mol):
    atom_feat = [atom_features(atom) for atom in mol.GetAtoms()]

    edge_attr = []
    edge_index = []

    for bond in mol.GetBonds():
        # eid = bond.GetIdx()
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([(i, j), (j, i)])
        b = bond_features(bond)
        edge_attr.extend([b, b.copy()])

    x = torch.FloatTensor(atom_feat)
    edge_attr = torch.FloatTensor(edge_attr)
    edge_index = torch.LongTensor(edge_index).T

    return Data(x=x, edge_attr=edge_attr, edge_index=edge_index)


def smiles2data(smi, explicit_h=True):
    mol = Chem.MolFromSmiles(smi)
    if explicit_h:
        mol = Chem.AddHs(mol)
    return mol2data(mol)


# from
# https://github.com/chemprop/chemprop/blob/master/chemprop/features/featurization.py

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    "atomic_num": list(range(MAX_ATOMIC_NUM)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

BOND_FDIM = 14


def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features(atom):
    features = (
        onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
        + onek_encoding_unk(
            int(atom.GetHybridization()), ATOM_FEATURES["hybridization"]
        )
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )  # scaled to about the same range as other features
    return features


def bond_features(bond):
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0),
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


# from
# https://github.com/chemprop/chemprop/blob/master/chemprop/nn_utils.py


def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.
    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: List[Union[float, int]],
        total_epochs: List[int],
        steps_per_epoch: int,
        init_lr: List[float],
        max_lr: List[float],
        final_lr: List[float],
    ):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        assert (
            len(optimizer.param_groups)
            == len(warmup_epochs)
            == len(total_epochs)
            == len(init_lr)
            == len(max_lr)
            == len(final_lr)
        )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (
            1 / (self.total_steps - self.warmup_steps)
        )

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.
        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.
        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = (
                    self.init_lr[i] + self.current_step * self.linear_increment[i]
                )
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (
                    self.exponential_gamma[i]
                    ** (self.current_step - self.warmup_steps[i])
                )
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]["lr"] = self.lr[i]
