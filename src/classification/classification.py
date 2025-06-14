import torch
import torch.nn as nn
import torch.optim as optim
from molecule_gnn_model import GNN, GNN_graphpred
from torch_geometric.data import DataLoader
from molecule_datasets import MoleculeDataset
from util import get_num_task
from sklearn.metrics import roc_auc_score
from config import args
from splitters import random_split
from os.path import isfile
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap 

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

def get_predictions_and_labels(model, device, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(((batch.y.view(-1).cpu().numpy() + 1) / 2).astype(int))
            y_pred.extend(preds)
    return y_true, y_pred

def main():
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    numTasks = get_num_task(args.dataset)
    dataset = MoleculeDataset("C:/Users/emirb/molecule_datasets/my_augmented", dataset="augmented.csv")
    print(f"Loaded {len(dataset)} molecules from {args.dataset}")
    trainDataset, valDataset, testDataset = random_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.runseed
    )

    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valLoader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testLoader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    moleculeModel = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    model = GNN_graphpred(args=args, num_tasks=numTasks, molecule_model=moleculeModel).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    if getattr(args, "C:/Users/emirb/pretraining_model.pth", "") and isfile(args.input_model_file):
        model.from_pretrained(args.input_model_file)
        for param in model.gnn.parameters():
            param.requires_grad = False

    for epoch in range(args.epochs):
        trainLoss = train(model, device, trainLoader, optimizer, criterion)
        valAcc = evaluate(model, device, valLoader)
        print(f"Epoch {epoch + 1}: Train loss {trainLoss:.4f}, Val accuracy {valAcc:.4f}")

    testAcc = evaluate(model, device, testLoader)
    print(f"Test accuracy: {testAcc:.4f}")

    y_true, y_pred = get_predictions_and_labels(model, device, testLoader)
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    orange_cmap = LinearSegmentedColormap.from_list("custom_orange", ["#fff5e6", "#ff9900", "#ff4400"])  # <-- Custom colormap
    display.plot(cmap=orange_cmap)
    plt.title("Confusion Matrix (Test Set)")
    plt.show()

if __name__ == '__main__':
    main()