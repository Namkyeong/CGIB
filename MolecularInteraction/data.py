import torch
torch.set_num_threads(2)

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

import random
random.seed(0)

import pandas as pd
import numpy as np
np.random.seed(0)

from torch_geometric.data import Data
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc

from tqdm import tqdm

from utils import one_of_k_encoding_unk, one_of_k_encoding


# Code is borrowed from CIGIN
def get_atom_features_mnsol(atom, stereo, features, explicit_H=False):
    """
    Method that computes atom level features from rdkit atom object
    :param atom:
    :param stereo:
    :param features:
    :param explicit_H:
    :return: the node features of an atom
    """
    possible_atoms = ['B', 'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D])
    atom_features += [int(i) for i in list("{0:06b}".format(features))]

    if not explicit_H:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except Exception as e:

        atom_features += [False, False
                          ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(atom_features)


def get_atom_features(atom, stereo, features, explicit_H=False):
    """
    Method that computes atom level features from rdkit atom object
    :param atom:
    :param stereo:
    :param features:
    :param explicit_H:
    :return: the node features of an atom
    """
    possible_atoms = ['H', 'B', 'C', 'N', 'O', 'F', 'Na', 'Si', 'P', 'S', 'Cl', 'Ge', 'Se', 'Br', 'Sn', 'Te', 'I']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms) # 17
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3]) # 4
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1]) # 2
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) # 7
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1]) # 3
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D]) # 5
    atom_features += [int(i) for i in list("{0:06b}".format(features))]

    if not explicit_H:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except Exception as e:

        atom_features += [False, False
                          ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(atom_features)


def get_bond_features(bond):
    """
    Method that computes bond level features from rdkit bond object
    :param bond: rdkit bond object
    :return: bond features, 1d numpy array
    """

    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats)



def get_graph_from_smile(molecule_smile, idx):
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    :param molecule_smile: SMILE sequence
    :return: DGL graph object, Node features and Edge features
    """

    molecule = Chem.MolFromSmiles(molecule_smile)
    features = rdDesc.GetFeatureInvariants(molecule)

    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0] * molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    node_features = []
    edge_features = []
    bonds = []
    for i in range(molecule.GetNumAtoms()):

        atom_i = molecule.GetAtomWithIdx(i)

        atom_i_features = get_atom_features(atom_i, chiral_centers[i], features[i])
        # Use get_atom_features_mnsol if you are building dataset related to solvation free energies
        # atom_i_features = get_atom_features_mnsol(atom_i, chiral_centers[i], features[i])
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    atom_feats = torch.tensor(node_features, dtype = torch.float)
    edge_index = torch.tensor(bonds, dtype = torch.long).T
    edge_feats = torch.tensor(edge_features, dtype = torch.float)

    return Data(x = atom_feats, edge_index = edge_index, edge_attr = edge_feats, idx = idx)


def build_dataset(dataset, graph1, graph2, target):

    processed = list()

    count = 0

    for idx in tqdm(range(len(dataset))):

        solute = dataset.loc[idx][graph1]
        mol = Chem.MolFromSmiles(solute)
        mol = Chem.AddHs(mol)
        solute = Chem.MolToSmiles(mol)
        solute_graph = get_graph_from_smile(solute, idx)

        solvent = dataset.loc[idx][graph2]
        mol = Chem.MolFromSmiles(solvent)
        mol = Chem.AddHs(mol)
        solvent = Chem.MolToSmiles(mol)
        solvent_graph = get_graph_from_smile(solvent, idx)
        
        delta_g = dataset.loc[idx][target]
        # delta_g = np.log(dataset.loc[idx][target])

        processed.append([solute_graph, solvent_graph, delta_g])

    return processed, dataset


class Dataclass(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        return self.dataset[idx]


if __name__ == "__main__":

    df = pd.read_csv(f'data/raw_data/chr_abs.csv', sep=",")
    # Chr : (Chromophore, Solvent, Absorption max (nm)/ Emission max (nm)/ Lifetime (ns))
    processed_data, dataset = build_dataset(df, "Chromophore", "Solvent", "Absorption max (nm)")
    torch.save(processed_data, "./data/processed/chr_abs.pt")
