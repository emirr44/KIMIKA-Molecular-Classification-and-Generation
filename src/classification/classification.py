from os.path import dirname, join
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from molecule_gnn_model import GNN, GNN_graphpred
from torch_geometric.data import DataLoader
from molecule_datasets import MoleculeDataset
from util import get_num_task
from sklearn.metrics import roc_auc_score
from config import args

def train(model, device, loader, optimizer, criterion):
    model.train()
    total = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float32)
        y = (y + 1) / 2
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, device, loader):
    model.eval()
    yTrue, yPred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            yTrue.append(batch.y.view(-1).cpu())
            yPred.append(torch.sigmoid(pred.view(-1)).cpu())
    yTrue = torch.cat(yTrue).numpy()
    yPred = torch.cat(yPred).numpy()
    yTrue = (yTrue + 1) / 2
    try:
        auc = roc_auc_score(yTrue, yPred)
    except ValueError:
        auc = float('nan')
    return auc

def main():
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    numTasks = get_num_task(args.dataset)
    dataset = MoleculeDataset(args.data_file, dataset=args.dataset)

    n = len(dataset)
    trainSize = int(0.8 * n)
    valSize = int(0.1 * n)
    indices = np.random.permutation(n)
    trainIdx, valIdx, testIdx = indices[:trainSize], indices[trainSize:trainSize+valSize], indices[trainSize+valSize:]
    trainDataset = dataset[trainIdx.tolist()]
    valDataset = dataset[valIdx.tolist()]
    testDataset = dataset[testIdx.tolist()]

    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valLoader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testLoader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    moleculeModel = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    model = GNN_graphpred(args=args, num_tasks=numTasks, molecule_model=moleculeModel).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        trainLoss = train(model, device, trainLoader, optimizer, criterion)
        valAcc = evaluate(model, device, valLoader)
        print(f"Epoch {epoch}: Train loss {trainLoss:.4f}, Val accuracy {valAcc:.4f}")

    testAcc = evaluate(model, device, testLoader)
    print(f"Test accuracy: {testAcc:.4f}")

if __name__ == '__main__':
    main()