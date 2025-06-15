# HIV Classification - how to train the model

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
which will require installing the ***preprocessing*** library into Colab.

## Step 2: The problem with importing *getGraphEmbeddings*

The ***processing*** library is obsolete when trying to install it on Colab. For that reason, it's better to use a direct implementation of *getGraphEmbeddings* function with 3 parameters. The implementation can be found in the `processing.py` file. Paste *getGraphEmbeddings* before pasting the code from `model.py`.

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

After finding and cleaning the dataset, from the *clean_hiv* function in `hiv_classification.py`

The classification will be done on the **HIV_active** column, using `hiv_cleaned.csv` and `gnn_embeddings.pt`.





