import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pickle
import random
import shutil

# --- Helper Functions ---
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} molecules.")
    return df

def randomize_smiles(smiles, n=5):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    randomized = set()
    for _ in range(n * 2):
        new_smiles = Chem.MolToSmiles(mol, doRandom=True)
        randomized.add(new_smiles)
        if len(randomized) >= n:
            break
    return list(randomized)

def enumerate_tautomers(smiles, max_tautomers=3):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    enumerator = rdMolStandardize.TautomerEnumerator()
    tautomers = enumerator.Enumerate(mol)
    unique_smiles = set()
    for t in tautomers:
        s = Chem.MolToSmiles(t, canonical=True)
        unique_smiles.add(s)
        if len(unique_smiles) >= max_tautomers:
            break
    return list(unique_smiles)

def is_valid(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def smiles_to_graph(smiles):
    # Placeholder: replace with your actual implementation
    return None

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

# --- Augmentation ---

df = load_dataset('bace.csv')

unique_augmented = set()
augmented_rows = []

for original_smiles in df['mol']:
    if original_smiles not in unique_augmented:
        augmented_rows.append((original_smiles, original_smiles, 'original'))
        unique_augmented.add(original_smiles)

    for rand_smiles in randomize_smiles(original_smiles, n=5):
        if is_valid(rand_smiles) and rand_smiles not in unique_augmented:
            augmented_rows.append((original_smiles, rand_smiles, 'random_smiles'))
            unique_augmented.add(rand_smiles)

    for taut in enumerate_tautomers(original_smiles, max_tautomers=3):
        if is_valid(taut) and taut not in unique_augmented:
            augmented_rows.append((original_smiles, taut, 'tautomer'))
            unique_augmented.add(taut)

aug_df = pd.DataFrame(augmented_rows, columns=['original_smiles', 'augmented_smiles', 'augmentation_type'])
aug_df.to_csv('augmented_bace_smiles.csv', index=False)

print("SMILES Augmentation Complete:")
print(f"Originals: {len(df)}")
print(f"Augmented (total): {len(aug_df)}")
print(aug_df['augmentation_type'].value_counts())

# --- Graph Conversion ---

aug_graphs = []
failed_smiles = []

for s in aug_df['augmented_smiles']:
    g = smiles_to_graph(s)
    if g is not None:
        aug_graphs.append(g)
    else:
        failed_smiles.append(s)

torch.save(aug_graphs, 'augmented_bace_graphs.pt')
pd.DataFrame(failed_smiles, columns=["failed_smiles"]).to_csv("failed_augmented_smiles.csv", index=False)

print("Graph Conversion Complete:")
print(f"Total SMILES input: {len(aug_df)}")
print(f"Successfully converted: {len(aug_graphs)}")
print(f"Failed to convert: {len(failed_smiles)}")

# --- Scaffold Split ---

aug_df['scaffold'] = aug_df['original_smiles'].apply(get_scaffold)
scaffold_to_smiles = defaultdict(list)
for orig, scaf in zip(aug_df['original_smiles'], aug_df['scaffold']):
    scaffold_to_smiles[scaf].append(orig)
scaffold_sets = [list(set(s)) for s in scaffold_to_smiles.values()]
random.seed(42)
random.shuffle(scaffold_sets)
train, valtest = train_test_split(scaffold_sets, test_size=0.2, random_state=42)
val, test = train_test_split(valtest, test_size=0.5, random_state=42)
train_smiles = set(s for group in train for s in group)
val_smiles = set(s for group in val for s in group)
test_smiles = set(s for group in test for s in group)

def assign_split(orig_smiles):
    if orig_smiles in train_smiles:
        return 'train'
    elif orig_smiles in val_smiles:
        return 'val'
    elif orig_smiles in test_smiles:
        return 'test'
    return 'unknown'

aug_df['split'] = aug_df['original_smiles'].apply(assign_split)

print("Split Summary:")
print(aug_df['split'].value_counts())

train_idx = aug_df[aug_df['split'] == 'train'].index.tolist()
val_idx = aug_df[aug_df['split'] == 'val'].index.tolist()
test_idx = aug_df[aug_df['split'] == 'test'].index.tolist()

with open('aug_train_idx.pkl', 'wb') as f:
    pickle.dump(train_idx, f)
with open('aug_val_idx.pkl', 'wb') as f:
    pickle.dump(val_idx, f)
with open('aug_test_idx.pkl', 'wb') as f:
    pickle.dump(test_idx, f)

print("Unique splits:", aug_df['split'].unique())
grouped = aug_df.groupby('original_smiles')['split'].nunique()
leaked = grouped[grouped > 1]
print(f"Leaking original SMILES: {len(leaked)}")
print("Split distribution:")
print(aug_df['split'].value_counts())
mismatch = aug_df[aug_df['split'] != aug_df.groupby('original_smiles')['split'].transform('first')]
print(f"Mismatched augmentations: {len(mismatch)}")
print("Scaffold NaNs:", aug_df['scaffold'].isna().sum())

aug_df.to_csv('augmented_bace_smiles.csv', index=False)
print("Saved: augmented_bace_smiles.csv")

shutil.make_archive('bace_data_aug', 'zip', './') 
