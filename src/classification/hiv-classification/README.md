# HIV Classification - how to train the model on the HIV dataset

First and foremost, get the `hiv.csv` dataset from DeepChem by typing the following commands into the terminal:
```
wget -O hiv.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/hiv.csv
mkdir -p ./molecule_datasets/hiv/raw
mv hiv.csv ./molecule_datasets/hiv/raw/hiv.csv
```

## Step 1: Moving everything to Google Colab

Because of the lack of resources to train it locally, the model was trained in Google Colab and `molecule_datasets.py`, `molecule_gnn_model.py` (the two files are from [here](https://github.com/chao1224/GraphMVP)), and `model.py` were pasted into the notebook with `model.py` being pasted last as this file is the one which executes everything. 

In `model.py` the following two lines indicate that there are two specific classes (*GNN_graphpredComplete* and *GNNComplete*) that are needed from `molecule_gnn_model.py` and one function (*mol_to_graph_data_obj_simple*) from `molecule_datasets.py`
```
from molecule_gnn_model import GNN_graphpredComplete, GNNComplete
from molecule_datasets import mol_to_graph_data_obj_simple
```

When running the part of code from `model.py` after pasting this into the notebook, this line will give error:
```
from preprocessing import getGraphEmbeddings
```
which will require installing the ***preprocessing*** library into Colab. However, we just used the direct implementation of the function we need.

## Step 2: The problem with importing *getGraphEmbeddings*

The ***preprocessing*** library is obsolete when trying to install it on Colab. For that reason, it's better to use a direct implementation of *getGraphEmbeddings* function with 3 parameters. The implementation can be found in the `preprocessing.py` file in the repository (**helper-functions** folder). Paste *getGraphEmbeddings* before pasting the code from `model.py`.

## Step 3: Cleaning the dataset

After running the last cell from `model.py`, there will be the following output:

OUTPUT:

>[13:00:14] Explicit valence for atom # 3 Al, 6, is greater than permitted  
>[13:00:14] Explicit valence for atom # 5 B, 5, is greater than permitted  
>[13:00:28] Explicit valence for atom # 16 Al, 9, is greater than permitted  
>[13:00:34] Explicit valence for atom # 4 Al, 9, is greater than permitted  
>[13:00:46] Explicit valence for atom # 12 Al, 7, is greater than permitted  
>[13:00:46] Explicit valence for atom # 13 Al, 7, is greater than permitted  
>[13:00:49] WARNING: not removing hydrogen atom without neighbors  
>[13:00:49] WARNING: not removing hydrogen atom without neighbors  
>[13:00:50] Explicit valence for atom # 6 Ge, 5, is greater than permitted  
>Loaded 41120 molecules.  
>Number of molecules that failed: 7

Out of 41127 instances in `hiv.csv`, 7 of them failed and because of that, the vector embeddings (`gnn_embeddings.pt`) consisted of 41120 instances. Since `hiv.csv` and `gnn_embeddings.pt` had different amounts of molecules, it was easier to find and remove the 7 molecules that failed to be converted and to make a new `hiv_cleaned.csv` file that exempt those 7 molecules from the new file. The new cleaned file, with 41120 molecules will then be used alongside the vector embeddings in classification tasks.

| File              | Instances |
| ----------------- | ----------|
| hiv.csv           | 41127     |
| gnn_embeddings.pt | 41120     |
| hiv_cleaned.csv   | 41120     |

The classification will be done on the **HIV_active** column, using `hiv_cleaned.csv` and `gnn_embeddings.pt`. After finding and cleaning the dataset, from the *clean_hiv* function in `hiv_classification.py`, the following 7 molecules will be removed

>Found 7 failed molecules:  
>1. O=C1O[Al]23(OC1=O)(OC(=O)C(=O)O2)OC(=O)C(=O)O3  
>2. Cc1ccc([B-2]2(c3ccc(C)cc3)=NCCO2)cc1  
>3. Oc1ccc(C2Oc3cc(O)cc4c3C(=[O+][AlH3-3]35([O+]=C6c7c(cc(O)cc7[OH+]3)OC(c3ccc(O)cc3O)C6O)([O+]=C3c6c(cc(O)cc6[OH+]5)OC(c5ccc(O)cc5O)C3O)[OH+]4)C2O)c(O)c1  
>4. CC1=C2[OH+][AlH3-3]34([O+]=C2C=CN1C)([O+]=C1C=CN(C)C(C)=C1[OH+]3)[O+]=C1C=CN(C)C(C)=C1[OH+]4  
>5. CC(c1cccs1)=[N+]1[N-]C(N)=[S+][AlH3-]12[OH+]B(c1ccccc1)[OH+]2  
>6. CC(c1ccccn1)=[N+]1[N-]C(N)=[S+][AlH3-]12[OH+]B(c1ccccc1)[OH+]2  
>7. [Na+].c1ccc([SH+][GeH2+]2[SH+]c3ccccc3[SH+]2)c([SH+][GeH2+]2[SH+]c3ccccc3[SH+]2)c1  

>Saved cleaned dataset to 'hiv_cleaned.csv'  
>Original count: 41127, Cleaned count: 41120

and the dataset is ready to be handled.

## Step 4: Classification

In `hiv_classification.py` two types of classification tasks are done: a Random Forest Classifier evaluating 6 metrics, including a confusion matrix, and a simple neural network. The results of these classification tasks are in the [main README](https://github.com/emirr44/QUARK-Molecule-Generating-AI-Model) of this repository.







