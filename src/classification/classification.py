import sys
sys.path.append(r"C:\Users\emirb\OneDrive\Documents\GitHub\GraphMVP\src_regression\models_complete_feature")

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from molecule_gnn_model import GNN, GNN_graphpred
from molecule_datasets import MoleculeDataset
from util import get_num_task
from config import args


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, device, loader, optimizer, criterion, scaler=None):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = ((batch.y.view(logits.shape) + 1) / 2).to(torch.float)
            loss = criterion(logits, target).mean()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def evaluate(model, device, loader, return_prob=False):
    model.eval()
    all_true, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(logits.view(-1))
            all_probs.append(probs.cpu().numpy())
            all_true.append(((batch.y.view(-1).cpu().numpy() + 1) / 2).astype(int))
    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_probs)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    if return_prob:
        return auc, acc, y_true, y_pred, y_prob
    else:
        return auc, acc, y_true, y_pred


def main():
    set_seed(args.runseed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load data using original parameters
    num_tasks = get_num_task(args.dataset)
    dataset = MoleculeDataset(
        "C:/Users/emirb/molecule_datasets/my_augmented", dataset="augmented.csv"
    )
    print(f"Loaded {len(dataset)} molecules from {args.dataset}")

    # Prepare stratified splits based on args.frac_train, frac_valid, frac_test
    labels = np.array([int(((d.y.item() + 1) / 2)) for d in dataset])
    temp_size = args.frac_valid + args.frac_test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_size, random_state=args.runseed)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(labels)), labels))
    temp_labels = labels[temp_idx]
    val_frac = args.frac_valid / temp_size
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_frac, random_state=args.runseed)
    val_rel, test_rel = next(sss2.split(np.zeros(len(temp_labels)), temp_labels))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Build model and optim
    backbone = GNN(
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type
    )
    model = GNN_graphpred(args=args, num_tasks=num_tasks, molecule_model=backbone).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and args.use_amp else None

    best_val_auc, patience = 0.0, 0
    if args.input_model_file and os.path.isfile(args.input_model_file):
        model.from_pretrained(args.input_model_file)
        for p in model.gnn.parameters(): p.requires_grad = False

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, device, train_loader, optimizer, criterion, scaler)
        val_auc, val_acc, _, _ = evaluate(model, device, val_loader)
        scheduler.step(val_auc)
        print(f"Epoch {epoch}/{args.epochs}: loss {loss:.4f}, val AUC {val_auc:.4f}, val ACC {val_acc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc, patience = val_auc, 0
            torch.save(model.state_dict(), args.save_model_file)
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print("Early stopping triggered.")
                break
    model.load_state_dict(torch.load(args.save_model_file, map_location=torch.device('cpu')))
    test_auc, test_acc, y_true, y_pred, y_prob = evaluate(model, device, test_loader, return_prob=True)
    print(f"Test AUC: {test_auc:.4f}, Test ACC: {test_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    cmap = LinearSegmentedColormap.from_list("custom_orange", ["#fff5e6", "#ff9900", "#ff4400"])
    disp.plot(cmap=cmap)
    plt.title("Confusion Matrix (Test Set)")
    plt.show()

    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
