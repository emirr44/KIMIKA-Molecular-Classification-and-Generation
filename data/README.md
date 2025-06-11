# Dataset

Both classification and molecule generation codes in this project rely on same dataset, bace.csv. The BACE dataset focuses on inhibitors of human beta-secretase 1 (BACE-1). It includes both quantitative (IC50 values) and qualitative (binary labels) binding results.

# Download Instructions

```bash
wget -O bace.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv
mkdir -p ./molecule_datasets/bace/raw
mv bace.csv ./molecule_datasets/bace/raw/bace.csv
```

Since mentioned dataset is realtively small (~1513 molecules), for successful training, and to prevent overfitting, data augmentation was necessary. 

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
>>> data
Data(x=[21, 6], edge_index=[2, 44], edge_attr=[44, 3])
```

This example shows a molecule with:

* **21 atoms**
* **22 bonds** (44 directional edges)
* **6 atom features**
* **3 bond features**

---

## File Summary

| File Name                   | Format  | Description                                     |
| --------------------------- | ------- | ----------------------------------------------- |
| `augmented_bace_smiles.csv` | CSV     | All original + augmented SMILES                 |
| `baseline_graphs.pt`        | PyTorch | Graphs from original (unaugmented) molecules    |
| `augmented_bace_graphs.pt`  | PyTorch | Graphs for all augmented SMILES                 |
| `aug_train_idx.pkl`         | Pickle  | Indices of augmented graphs in training split   |
| `aug_val_idx.pkl`           | Pickle  | Indices of augmented graphs in validation split |
| `aug_test_idx.pkl`          | Pickle  | Indices of augmented graphs in test split       |



You can access this file [here](https://drive.google.com/drive/folders/1e8gsnhkNpFcfQnZoSztilspu0xJBnH0L?usp=sharing)


---

## Dataset Statistics

| Type             | Count  |
| ---------------- | ------ |
| Original SMILES  | 1,513  |
| Augmented SMILES | 12,721 |
| Total Graphs     | 12,721 |
| Train Graphs     | 10,759 |
| Val Graphs       | 995    |
| Test Graphs      | 967    |

---

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
