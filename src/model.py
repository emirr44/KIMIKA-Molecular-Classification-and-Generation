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
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from os.path import join

regCriterion = nn.MSELoss()

def train(model, device, loader, optimizer):
    model.train()
    totalLoss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        predictions = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
        y = batch.y.squeeze()
        loss = regCriterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totalLoss += loss.detach().item()
    return totalLoss / len(loader)

def evaluate(model, device, loader):
    model.evaluate()
    yTrue, yPredicted = [], []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(1)
        true = batch.y.squeeze()
        yTrue.append(true)
        yPredicted.append(pred)
    yTrue = torch.cat(yTrue, dim= 0).cpu().numpy()
    yPredicted = torch.cat(yPredicted, dim=0).cpu().numpy()
    rmse = mean_squared_error(yTrue, yPredicted, squared=False)
    mae = mean_absolute_error(yTrue, yPredicted)
    return {'RMSE': rmse, 'MAE': mae}, yTrue, yPredicted

def saveModel(model, moleculeModel, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'molecule_model': moleculeModel.state_dict(),
        'model': model.state_dict()
    }, path)

def printMetrics(epoch, loss, train, metrics):
    print(f"Epoch {epoch}, Loss: {loss:.6f}")
    for metric in metrics:
        print(f"{metric}: Train={train[metric]:.4f}")
    print()

def smilesToGraph(path, smiles_col='mol', label_col='Class'):
    df = pd.read_csv(path)
    smiles = df[smiles_col]
    labels = df[label_col]
    dataList = []
    for i, (smiles, label) in enumerate(zip(smiles, labels)):
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            continue
        data = mol_to_graph_data_obj_simple(mol)
        data.id = torch.tensor([i])
        data.y = torch.tensor([label])
        dataList.append(data)
    return dataList

def getGraphEmbeddings(model, device, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    embeddings = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            emb = model.molecule_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(0)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    numTasks = 1
    path = 'path to dataset'
    dataset = smilesToGraph(path, smiles_col='mol', label_col='Class')
    print(f"Loaded {len(dataset)} molecules.")
    
    trainLoader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    moleculeModel = GNNComplete(num_layer=5, emb_dim=300, JK='last', drop_ratio=0.5, gnn_type="gin")
    model = GNN_graphpredComplete(graph_pooling="mean", num_tasks=numTasks, molecule_model=moleculeModel)    

    inputModelFile = 'file with pretrained weights'
    if os.path.exists(inputModelFile):
        model.from_pretrained(inputModelFile)
    model.to(device)
    print(model)

    modelParameterGroup = [
        {'params' : model.molecule_model.parameters()},
        {'params' : model.graph_pred_linear.parameters(), "lr": 0.001 * 1}
    ]
    optimizer = optim.Adam(modelParameterGroup, lr=0.001, weight_decay=0)

    outputDirectory = 'output directory location'
    metric_list = ["RMSE", "MAE"]
    trainHist = []

    for epoch in range(10):
        loss = train(model, device, trainLoader, optimizer)
        print(f"Epoch: {epoch}, Loss: {loss}")
        trainResult, _, _  = eval(model, device, trainLoader)
        trainHist.append(trainResult)

        printMetrics(epoch, loss, trainResult, metric_list)
    
    saveModel(model, moleculeModel, join(outputDirectory, 'model_final.pth'))

    embeddings = getGraphEmbeddings(model, device, dataset)
    torch.save(embeddings, join(outputDirectory, 'gnn_embeddings.pt'))
    np.save(join(outputDirectory, 'gnn_embeddings.npy'), embeddings.numpy())

if __name__ == "__main__":
    main()