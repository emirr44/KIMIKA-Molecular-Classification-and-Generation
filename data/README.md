# Dataset

We use the BACE dataset from MoleculeNet, which contains small-molecule inhibitors of human β-secretase 1

Key points:
- **Data content** - 1,513 unique molecules (SMILES) with binary labels (1 = active inhibitor, 0 = inactive). The raw data also includes IC50 values, but we use only the binary labels for classification.
- **Importance** - BACE is a validated drug target in Alzheimer’s research. Inhibiting BACE can potentially reduce amyloid plaque formation
- **Preprocessing** - We convert SMILES strings to graph representations. Each atom is featurized by its element type, chirality, and other one-hot features; each bond is encoded by type and aromaticity.
- **Train/Val/Test split** - We adopt a scaffold split to simulate realistic conditions (training, validation, and test share no common scaffolds) Scaffold splitting, according to the MoleculeNet protocol for molecular property prediction, is recommended for BACE since it enforces domain generalization.

# Download Instructions

```bash
wget -O bace.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv
mkdir -p ./molecule_datasets/bace/raw
mv bace.csv ./molecule_datasets/bace/raw/bace.csv
```

Since mentioned dataset is realtively small (~1513 molecules), for successful training, and to prevent overfitting, data augmentation was necessary. You can access file with augmented dataset, baseline graphs and more [here](https://drive.google.com/drive/folders/1e8gsnhkNpFcfQnZoSztilspu0xJBnH0L?usp=sharing)


---

## Graph Data Format

Each molecule is converted into a `torch_geometric.data.Data` object representing a graph:

- `x`: Node feature matrix — one row per atom.
- `edge_index`: Connectivity matrix — stores bidirectional edges.
- `edge_attr`: Edge feature matrix — one row per bond (bidirectional).

### Atom Features (`x`)
| Feature               | Description                        |
|-----------------------|------------------------------------|
| Atomic number         | Integer ID for each element        |
| Degree                | Number of connected atoms          |
| Formal charge         | Net atomic charge                  |
| Hybridization (enum)  | sp, sp2, sp3...                    |
| Total Hydrogens       | Number of H atoms                  |
| Is Aromatic           | Boolean (0 or 1)                   |

### Bond Features (`edge_attr`)
| Feature               | Description                        |
|-----------------------|------------------------------------|
| Bond type             | Single, double, triple             |
| Is conjugated         | Boolean                            |
| Is in ring            | Boolean                            |


### Example Graph Object

```python
Data(x=[21, 6], edge_index=[2, 44], edge_attr=[44, 3])
```

This example shows a molecule with:

* **21 atoms**
* **22 bonds** (44 directional edges)
* **6 atom features**
* **3 bond features**


## Dataset Statistics

| Type             | Count  |
| ---------------- | ------ |
| Original SMILES  | 1,513  |
| Augmented SMILES | 12,721 |
| Total Graphs     | 12,721 |
| Train Graphs     | 10,759 |
| Val Graphs       | 995    |
| Test Graphs      | 967    |


## Data Flow Diagram

```text
Raw SMILES (bace.csv)
        ↓
[Cleaned + Validated]
        ↓
+-------------------------------+
|  Augmentation (random, taut) |
+-------------------------------+
        ↓
augmented_bace_smiles.csv
        ↓
smiles_to_graph()
        ↓
augmented_bace_graphs.pt  <--- Indexed using aug_train_idx.pkl, etc.
```
