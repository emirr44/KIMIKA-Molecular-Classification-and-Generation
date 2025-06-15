import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import DataLoader
from molecule_gnn_model import GNN_graphpredComplete, GNNComplete
from molecule_datasets import mol_to_graph_data_obj_simple
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from os.path import join
from preprocessing import getGraphEmbeddings

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    csv_path = 'C:/Users/emirb/molecule_datasets/hiv/raw/hiv.csv'
    df = pd.read_csv(csv_path)

    smiles_list = df['smiles'].tolist()
    failed = 0
    dataset = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed += 1
            continue
        dataset.append(mol_to_graph_data_obj_simple(mol))

    print(f"Loaded {len(dataset)} molecules.")
    print(f"Number of molecules that failed: {failed}")
    
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    moleculeModel = GNNComplete(num_layer=5, emb_dim=300, JK='last', drop_ratio=0.5, gnn_type="gin")
    model = GNN_graphpredComplete(graph_pooling="mean", num_tasks=1, molecule_model=moleculeModel)

    inputModelFile = '/content/drive/MyDrive/pretraining_model_complete.pth' #indicate path to pretraining_model_complete.pth
    ############### Leave this if training locally ################
    if os.path.exists(inputModelFile):
        model.from_pretrained(inputModelFile)
    model.to(device)
    model.eval()
    ################################################################

    ############### Important for training in Colab ################
    '''
    if os.path.exists(inputModelFile):
        checkpoint = torch.load(inputModelFile, map_location='cpu', weights_only=True)

        # Handle atom encoder mismatch
        if 'atom_encoder.atom_embedding_list.1.weight' in checkpoint:
            if checkpoint['atom_encoder.atom_embedding_list.1.weight'].shape[0] == 4:
                print("Adapting pretrained weights from 4 to 5 atom types")
                checkpoint['atom_encoder.atom_embedding_list.1.weight'] = torch.cat([
                    checkpoint['atom_encoder.atom_embedding_list.1.weight'],
                    torch.randn(1, 300) * 0.02  # Small random initialization
                ], dim=0)

        model.load_state_dict(checkpoint, strict=False)
    '''
    #####################################################################################

    embeddings = getGraphEmbeddings(model, device, dataset)

    output_dir = 'C:/Users/emirb/OneDrive/Desktop/coding'
    os.makedirs(output_dir, exist_ok=True)
    torch.save(embeddings, join(output_dir, 'gnn_embeddings.pt'))
    np.save(join(output_dir, 'gnn_embeddings.npy'), embeddings.cpu().numpy())


if __name__ == "__main__":
    main()