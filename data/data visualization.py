import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sns.set(style="whitegrid")

def plot_smiles_histogram(df):
    df['smiles_len'] = df['mol'].apply(len)

    plt.hist(df['smiles_len'], bins=40, color='purple')
    plt.xlabel("SMILES Length")
    plt.ylabel("Count")
    plt.title("SMILES Length Distribution")
    plt.grid(True)
    plt.show()

def class_distribution_barplot(df, label_col='Class'):
    class_counts = df[label_col].value_counts().sort_index()
    class_names = class_counts.index.astype(str)

    plt.figure(figsize=(5, 4.5))
    plt.bar(class_names, class_counts, color=['#ff4400', "#5F0E85"])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class distribution (Inactive vs Active)")
    plt.grid(axis='y', linestyle='-', alpha=0.1)
    plt.show()

def plot_top_atom_types(df, smiles_col='mol', top_n=10):
    def extract_atom_symbols(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        return [atom.GetSymbol() for atom in mol.GetAtoms()]

    atom_counter = Counter()
    for s in df[smiles_col]:
        atom_counter.update(extract_atom_symbols(s))
    common_atoms = atom_counter.most_common(top_n)
    atoms, counts = zip(*common_atoms)

    plt.bar(atoms, counts, color='#ff4400')
    plt.xlabel("Atom Type")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} Atom Types in Dataset")
    plt.grid(axis='y', linestyle='--', alpha=0.2)
    plt.show()

def mol_weights_histogram(df):
    def safe_molwt(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Descriptors.MolWt(mol)
        return None
    df['MolWt'] = df['mol'].apply(safe_molwt)
    df_no_nan = df.dropna(subset=['MolWt'])

    plt.hist(df_no_nan['MolWt'], bins=50, color='steelblue')
    plt.xlabel("Molecular Weight")
    plt.ylabel("Count")
    plt.title("Molecular Weight Distribution")
    plt.grid(True)
    plt.show()

def plot_descriptor_correlation_heatmap(df):
    def calc_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None, None, None]
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumRotatableBonds(mol)
        ]
    desc = df['mol'].apply(calc_descriptors)
    desc_df = pd.DataFrame(desc.tolist(), columns=['MolWt', 'LogP', 'RotBonds'])
    desc_df = desc_df.dropna()
    corr = desc_df.corr()

    orange_cmap = LinearSegmentedColormap.from_list("custom_orange", ["#fff5e6", "#ff9900", "#ff4400"])
    sns.heatmap(corr, annot=True, cmap=orange_cmap)
    plt.title("Descriptor Correlation Heatmap")
    plt.show()
