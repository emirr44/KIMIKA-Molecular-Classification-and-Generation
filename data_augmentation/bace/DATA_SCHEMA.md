<details> <summary>Click to Expand the Markdown Content to Paste</summary># Data Schema: Augmented BACE Dataset

---

## 1. Input Files

### `bace.csv`
- **Columns**:
  - `mol` (string): Canonical SMILES of a BACE inhibitor.
  - `pIC50` (float): Bioactivity value (log IC<sub>50</sub>).
  - `Class` (int): Binary label (0 = inactive, 1 = active).

---

## 2. Augmentation Metadata

### `augmented_bace_smiles.csv`
| Column             | Description                                                      |
|--------------------|------------------------------------------------------------------|
| `original_smiles`  | Original (canonical) SMILES                                      |
| `augmented_smiles` | Augmented SMILES (random or tautomer)                            |
| `augmentation_type`| One of `original`, `random_smiles`, or `tautomer`                |
| `split`            | Assigned partition: `train`, `val`, or `test`                     |

---

## 3. Graph Files

### `baseline_graphs.pt`
- **Type**: `List[torch_geometric.data.Data]`
- **Size**: 1,513 entries (one per original SMILES)
- **Node Features (`x`)**:  
  - Shape: `[num_atoms, 6]`  
  - Columns:  
    1. Atomic number (int)  
    2. Atom degree (int)  
    3. Formal charge (int)  
    4. Hybridization (int code)  
    5. Number of implicit H atoms (int)  
    6. Is aromatic (0/1)  
- **Edge Index (`edge_index`)**:  
  - Shape: `[2, num_edges]`  
  - Each column → (source, target)  
- **Edge Attributes (`edge_attr`)**:  
  - Shape: `[num_edges, 3]`  
  - Columns:  
    1. Bond type (as double)  
    2. Is conjugated (0/1)  
    3. Is in ring (0/1)  

### `augmented_bace_graphs.pt`
- **Type**: `List[torch_geometric.data.Data]`
- **Size**: 12,721 entries (augmented + original)
- **Features**: Same schema as `baseline_graphs.pt`.
- **Index Alignment**: Position in this list corresponds to rows in `augmented_bace_smiles.csv`.

---

## 4. Split Index Files

- `aug_train_idx.pkl`  
- `aug_val_idx.pkl`  
- `aug_test_idx.pkl`  

Each is a pickled Python list of integer indices referencing entries in `augmented_bace_graphs.pt`.

---

## 5. Notes & Conventions

- All augmented SMILES were **validated** with RDKit’s `Chem.MolFromSmiles()`.
- No `NaN` or missing features exist in final graphs.
- 3D coordinates were **not** used in this version (position arrays absent).
- Hybridization codes and bond types follow RDKit enums.

---
</details>
