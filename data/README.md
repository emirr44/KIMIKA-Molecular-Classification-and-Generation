# Dataset

Both classification and molecule generation codes in this project rely on same dataset, bace.csv. The BACE dataset focuses on inhibitors of human beta-secretase 1 (BACE-1). It includes both quantitative (IC50 values) and qualitative (binary labels) binding results.

Download using this command:

```bash
wget -O bace.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv
mkdir -p ./molecule_datasets/bace/raw
mv bace.csv ./molecule_datasets/bace/raw/bace.csv
```

Since mentioned dataset is realtively small (~1513 molecules), for successful training, and to prevent overfitting, data augmentation was necessary. 

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
