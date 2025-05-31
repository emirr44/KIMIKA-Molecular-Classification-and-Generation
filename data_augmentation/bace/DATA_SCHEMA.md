---

````markdown
# DATA_SCHEMA.md

## 1. Overview

This document outlines the schema and structure of all files produced during the BACE data augmentation process for use with graph neural networks (GNNs).

---

## 2. Graph Data Format

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
````

This example shows a molecule with:

* **21 atoms**
* **22 bonds** (44 directional edges)
* **6 atom features**
* **3 bond features**

---

## 3. File Summary

| File Name                   | Format  | Description                                     |
| --------------------------- | ------- | ----------------------------------------------- |
| `augmented_bace_smiles.csv` | CSV     | All original + augmented SMILES                 |
| `baseline_graphs.pt`        | PyTorch | Graphs from original (unaugmented) molecules    |
| `augmented_bace_graphs.pt`  | PyTorch | Graphs for all augmented SMILES                 |
| `aug_train_idx.pkl`         | Pickle  | Indices of augmented graphs in training split   |
| `aug_val_idx.pkl`           | Pickle  | Indices of augmented graphs in validation split |
| `aug_test_idx.pkl`          | Pickle  | Indices of augmented graphs in test split       |

---

## 4. Dataset Statistics

| Type             | Count  |
| ---------------- | ------ |
| Original SMILES  | 1,513  |
| Augmented SMILES | 12,721 |
| Total Graphs     | 12,721 |
| Train Graphs     | 10,759 |
| Val Graphs       | 995    |
| Test Graphs      | 967    |

---

## 5. Data Flow Diagram

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

---

**Last Updated**: May 31, 2025
**Author**: Amir Sarajlić

```
