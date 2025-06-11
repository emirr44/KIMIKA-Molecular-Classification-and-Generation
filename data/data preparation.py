import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} molecules.")
    return df

def data_description(df):
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nNumerical Description:\n", df.describe(include=[np.number]))

    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Class Distribution ---")
    print(df['Class'].value_counts())

    print("\n--- Sample SMILES ---")
    print(df['mol'].head(10).tolist())

def duplicated(df):
    dup_rows = df.duplicated()
    print(f"Full duplicate rows: {dup_rows.sum()}")

    if 'smiles' in df.columns:
        dup_smiles = df['smiles'].duplicated()
        print(f"Duplicate SMILES: {dup_smiles.sum()}")

def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None

def validate_smiles(df):
    df_valid = df[df['valid_smiles']].copy()

    def strict_validate(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False

    df_valid = df_valid[df_valid['mol'].apply(strict_validate)]
    original_count = len(df)
    valid_count = len(df_valid)
    dropped_count = original_count - valid_count

    print(f"Original rows: {original_count}")
    print(f"Valid SMILES rows: {valid_count}")
    print(f"Dropped rows: {dropped_count}")

    df_dropped = df[~df.index.isin(df_valid.index)]
    df_dropped.to_csv("bace_dropped_invalid_smiles.csv", index=False)
    df_valid.reset_index(drop=True, inplace=True)
    return df_valid

def smiles_to_morgan_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def generate_and_save_fp(df, smiles_column='mol', output_file='baseline_fps.npy', radius=2, nBits=2048):
    df['morgan_fp'] = df[smiles_column].apply(smiles_to_morgan_fp)
    df = df[df['morgan_fp'].notna()].reset_index(drop=True)

    fingerprint_matrix = np.stack(df['morgan_fp'].values)

    np.save(output_file, fingerprint_matrix)
    print(f"Morgan fingerprint matrix shape: {fingerprint_matrix.shape}")
    return fingerprint_matrix, df

def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic())
    ], dtype=torch.float)

def bond_features(bond):
    return torch.tensor([
        int(bond.GetBondTypeAsDouble()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ], dtype=torch.float)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_feats)

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_index.append((i, j))
        edge_index.append((j, i))

        bf = bond_features(bond)
        edge_attr.append(bf)
        edge_attr.append(bf)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def convert_smiles_column_to_graphs(df, smiles_column='mol', output_file='baseline_graphs.pt'):
    graph_list = []
    for smiles in df[smiles_column]:
        g = smiles_to_graph(smiles)
        if g is not None:
            graph_list.append(g)
    torch.save(graph_list, output_file)
    print(f"✅ Converted {len(graph_list)} molecules into graphs.")
