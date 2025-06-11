import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import selfies as sf
from rdkit import Chem

def smiles_to_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except Exception:
        return None

def selfies_to_smiles(selfies):
    try:
        return sf.decoder(selfies)
    except Exception:
        return None

def build_selfies_vocab(selfies_list):
    charset = set()
    for s in selfies_list:
        charset.update(sf.split_selfies(s))
    charset.update(['[START]', '[END]'])
    idx2char = ['[nop]'] + sorted(charset)
    char2idx = {c: i for i, c in enumerate(idx2char)}
    return char2idx, idx2char

def selfies_to_tensor(selfies, char2idx, max_len):
    tokens = ['[START]'] + list(sf.split_selfies(selfies)) + ['[END]']
    idxs = [char2idx.get(tok, 0) for tok in tokens]
    idxs += [0] * (max_len - len(idxs))
    return torch.tensor(idxs[:max_len], dtype=torch.long)

def tensor_to_selfies(tensor, idx2char):
    tokens = [idx2char[i] for i in tensor if i != 0]
    if '[END]' in tokens:
        tokens = tokens[:tokens.index('[END]')]
    return ''.join(t for t in tokens if t not in ['[START]', '[END]', '[nop]'])

class SmilesVAE(nn.Module):
    def __init__(self, vocab_size, max_len, emb_dim=128, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        emb = self.emb(x)
        _, (h, _) = self.encoder(emb)
        h = h[-1]
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_input):
        h0 = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        emb = self.emb(seq_input)
        out, _ = self.decoder(emb, (h0, c0))
        return self.output(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x)
        return logits, mu, logvar

def vae_loss(logits, targets, mu, logvar, step=None, kl_anneal_steps=1000):
    recon_loss = nn.CrossEntropyLoss(ignore_index=0)(logits.view(-1, logits.size(-1)), targets.view(-1))
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_weight = min(1.0, step / kl_anneal_steps) if step is not None else 1.0
    return recon_loss + kl_weight * kld

def generate_molecules(vae, idx2char, char2idx, max_len, n_samples=10):
    device = next(vae.parameters()).device
    vae.eval()
    z = torch.randn(n_samples, vae.latent_to_hidden.in_features).to(device)
    input_seq = torch.full((n_samples, 1), char2idx['[START]'], dtype=torch.long).to(device)

    hidden = torch.tanh(vae.latent_to_hidden(z)).unsqueeze(0)
    cell = torch.zeros_like(hidden)

    sequences = [[] for _ in range(n_samples)]

    for _ in range(max_len):
        emb = vae.emb(input_seq[:, -1].unsqueeze(1))
        out, (hidden, cell) = vae.decoder(emb, (hidden, cell))
        logits = vae.output(out.squeeze(1))
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        input_seq = torch.cat([input_seq, next_token.unsqueeze(1)], dim=1)

        for i, token_id in enumerate(next_token.tolist()):
            sequences[i].append(token_id)

    selfies_out = []
    for seq in sequences:
        tokens = [idx2char[t] for t in seq]
        if '[END]' in tokens:
            tokens = tokens[:tokens.index('[END]')]
        selfies = ''.join([t for t in tokens if t not in ('[START]', '[END]', '[nop]')])
        selfies_out.append(selfies)

    return selfies_out

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('/content/drive/MyDrive/augmented_bace_smiles.csv')
    smiles_list = df['augmented_smiles'].tolist()[:1000]

    selfies_list = []
    for s in smiles_list:
        encoded = smiles_to_selfies(s)
        if encoded is not None:
            selfies_list.append(encoded)

    print(f"Loaded {len(selfies_list)} valid molecules.")

    max_len = max(len(list(sf.split_selfies(s))) + 2 for s in selfies_list)
    char2idx, idx2char = build_selfies_vocab(selfies_list)
    vocab_size = len(char2idx)

    tensor_data = [selfies_to_tensor(s, char2idx, max_len) for s in selfies_list]
    data = torch.stack(tensor_data)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=128, shuffle=True)

    vae = SmilesVAE(vocab_size, max_len).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    epochs = 10
    step = 0
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = vae(batch)
            loss = vae_loss(logits, batch, mu, logvar, step)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
            step += 1
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.4f}")

    selfies_gen = generate_molecules(vae, idx2char, char2idx, max_len, n_samples=10)
    smiles_gen = [selfies_to_smiles(s) for s in selfies_gen]

    print("Generated SMILES:")
    for s in smiles_gen:
        print(s)

    pd.DataFrame({'smiles': smiles_gen}).to_csv('/content/drive/MyDrive/generated_smiles.csv', index=False)

if __name__ == "__main__":
    main()
