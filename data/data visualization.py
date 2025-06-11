import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors

sns.set(style="whitegrid")

def plot_mol_weights_histogram(df):
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

def plot_smiles_histogram(df):
    df['smiles_len'] = df['mol'].apply(len)

    plt.hist(df['smiles_len'], bins=40, color='purple')
    plt.xlabel("SMILES Length")
    plt.ylabel("Count")
    plt.title("SMILES Length Distribution")
    plt.grid(True)
    plt.show()

def print_top_atom_types(df, smiles_col='mol', top_n=10):
    def extract_atom_symbols(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        return [atom.GetSymbol() for atom in mol.GetAtoms()]

    atom_counter = Counter()
    for s in df[smiles_col]:
        atom_counter.update(extract_atom_symbols(s))
    common_atoms = atom_counter.most_common(top_n)

    print(f"Top {top_n} Atom Types:")
    for atom, count in common_atoms:
        print(f"{atom}: {count}")
    return common_atoms

def plot_atom_types(common_atoms):
    atoms, counts = zip(*common_atoms)

    plt.bar(atoms, counts, color='green')
    plt.xlabel("Atom Type")
    plt.ylabel("Frequency")
    plt.title("Top 10 Atom Types in Dataset")
    plt.grid(True)
    plt.show()
