import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def cyclic_annealing_beta(epoch, cycle_length=10):
    return float((epoch % cycle_length) / cycle_length)

class SmilesTokenizer:
    def __init__(self, smiles_list):
        chars = sorted(set(''.join(smiles_list)))
        self.special_tokens = ['<PAD>', '<START>', '<END>']
        all_tokens = self.special_tokens + chars
        self.char2idx = {c: i for i, c in enumerate(all_tokens)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.pad_idx = self.char2idx['<PAD>']
        self.start_idx = self.char2idx['<START>']
        self.end_idx = self.char2idx['<END>']
        self.vocab_size = len(self.char2idx)

    def encode(self, s, max_len):
        s = '<START>' + s + '<END>'
        arr = np.full(max_len, self.pad_idx, dtype=int)
        for i, c in enumerate(s[:max_len]):
            arr[i] = self.char2idx.get(c, self.pad_idx)
        return arr

    def decode(self, arr):
        chars = []
        for i in arr:
            if i == self.end_idx:
                break
            if i > 2:  
                chars.append(self.idx2char.get(i, ''))
        return ''.join(chars)

class SmilesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len=100):
        self.smiles = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encoded = [tokenizer.encode(s, max_len) for s in smiles_list]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        return x, x

class SmilesVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=64, max_len=100, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_rnn = nn.LSTM(embed_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.dropout = nn.Dropout(p=0.3) 

    def encode(self, x):
        emb = self.embed(x)
        _, (h, _) = self.encoder_rnn(emb)
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq):
        emb = self.embed(seq)
        z_expand = z.unsqueeze(1).repeat(1, seq.size(1), 1)
        dec_input = torch.cat([emb, z_expand], dim=-1)
        out, _ = self.decoder_rnn(dec_input)
        logits = self.fc_out(out)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = self.dropout(z)
        logits = self.decode(z, x)
        return logits, mu, logvar

def vae_loss(logits, targets, mu, logvar, beta=1.0):
    recon_loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=0
    )
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl

df_bace = pd.read_csv('.../bace.csv') 
smiles_list = df_bace['mol'].tolist()

tokenizer = SmilesTokenizer(smiles_list)
dataset = SmilesDataset(smiles_list, tokenizer, max_len=120)

val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SmilesVAE(tokenizer.vocab_size, pad_idx=tokenizer.pad_idx).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = float('inf')
patience, trials = 10, 0
save_path = 'vae.pt'

for epoch in range(1, 101):
    model.train()
    total_loss = 0
    beta = cyclic_annealing_beta(epoch)

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, mu, logvar = model(x)
        loss, recon, kl = vae_loss(logits, y, mu, logvar, beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total_loss += loss.item()
    
    avg_train = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, mu, logvar = model(x)
            loss, _, _ = vae_loss(logits, y, mu, logvar, beta)
            val_loss += loss.item()
    avg_val = val_loss / len(val_loader)

    print(f"Epoch {epoch} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), save_path)
        trials = 0
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping triggered.")
            break

def generate_smiles(model, tokenizer, num_samples=16):
    model.eval()
    z = torch.randn(num_samples, model.latent_dim).to(device)
    sequences = torch.full((num_samples, 1), tokenizer.start_idx, dtype=torch.long).to(device)
    
    for _ in range(model.max_len):
        logits = model.decode(z, sequences)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        sequences = torch.cat([sequences, next_token], dim=1)
        if (next_token == tokenizer.end_idx).all():
            break

    sequences = sequences[:, 1:]  
    smiles = [tokenizer.decode(seq.cpu().numpy()) for seq in sequences]
    return smiles

model.load_state_dict(torch.load(save_path, map_location=device))

samples = generate_smiles(model, tokenizer, num_samples=16)
valids = [s for s in samples if Chem.MolFromSmiles(s)]
print(f"Valid molecules: {len(valids)} / {len(samples)}")

for i, smi in enumerate(samples, 1):
    print(f"Generated molecule {i}: {smi}")


model.eval()
latents = []
with torch.no_grad():
    for x, _ in DataLoader(dataset, batch_size=128):
        x = x.to(device)
        mu, _ = model.encode(x)
        latents.append(mu.cpu().numpy())
latents = np.vstack(latents)

pca = PCA(n_components=2)
z2 = pca.fit_transform(latents)
plt.figure()
plt.scatter(z2[:,0], z2[:,1])
plt.title("PCA of VAE Latent Space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
