import nn_preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset, DataLoader
import push_to_file
import random
import itertools

NUM_CHAMPIONS = 170
NUM_ROLES = 5
DRAFT_LENGTH = 20

INPUT_SIZE = NUM_CHAMPIONS * DRAFT_LENGTH
ROLE_NAMES = ["top", "jungle", "mid", "bot", "support"]

draft_df = nn_preprocessing.load_full_draft_data()  # all drafts dataframe
player_data = nn_preprocessing.load_player_data()  # player dict[str, pd.DataFrame]
pickban_df = nn_preprocessing.load_pickban_data()  # pickban dict[str, pd.DataFrame]
champ_matchup_info = nn_preprocessing.clean_matchup_info()  # dict[str, pd.DataFrame]

# Build champ-index mapping
all_champs = set()
for x in push_to_file.champs_list():
    all_champs.add(x)
champ2idx = {c: i+1 for i, c in enumerate(sorted(all_champs))}
idx2champ = {i: c for c, i in champ2idx.items()}
PAD_IDX = 0
VOCAB_SIZE = len(champ2idx) + 1  # include padding

# Encode draft into fixed-length sequence of champ indices
def encode_sequence(row):
    seq = []
    
    # phase1 bans
    seq.append(champ2idx.get(row.blue_fs_bans[0], PAD_IDX))
    seq.append(champ2idx.get(row.red_fs_bans[0], PAD_IDX))
    seq.append(champ2idx.get(row.blue_fs_bans[1], PAD_IDX))
    seq.append(champ2idx.get(row.red_fs_bans[1], PAD_IDX))
    seq.append(champ2idx.get(row.blue_fs_bans[2], PAD_IDX))
    seq.append(champ2idx.get(row.red_fs_bans[2], PAD_IDX))
    
    # phase1 picks (B1, R2, B2, R1)
    seq.append(champ2idx.get(row.blue_picks[0], PAD_IDX))
    seq.extend(champ2idx.get(x, PAD_IDX) for x in row.red_picks[:2])
    seq.extend(champ2idx.get(x, PAD_IDX) for x in row.blue_picks[1:3])
    seq.append(champ2idx.get(row.red_picks[2], PAD_IDX))
    
    # phase2 bans
    seq.append(champ2idx.get(row.red_ss_bans[0], PAD_IDX))
    seq.append(champ2idx.get(row.blue_ss_bans[0], PAD_IDX))
    seq.append(champ2idx.get(row.red_ss_bans[1], PAD_IDX))
    seq.append(champ2idx.get(row.blue_ss_bans[1], PAD_IDX))

    # phase2 picks (R1, B2, R1)
    seq.append(champ2idx.get(row.red_picks[3], PAD_IDX))
    seq.extend(champ2idx.get(x, PAD_IDX) for x in row.blue_picks[3:5])
    seq.append(champ2idx.get(row.red_picks[4], PAD_IDX))
    return seq

# Precompute pick/ban meta relevance per patch
meta_vectors = {}
for patch, df_patch in pickban_df.items():
    # detect columns
    pick_col = 'pick_rate' if 'pick_rate' in df_patch.columns else ('pick_pct' if 'pick_pct' in df_patch.columns else None)
    ban_col = 'ban_rate' if 'ban_rate' in df_patch.columns else ('ban_pct' if 'ban_pct' in df_patch.columns else None)
    vec = np.zeros(VOCAB_SIZE, dtype=float)
    if pick_col and ban_col:
        for champ, idx in champ2idx.items():
            r = df_patch[df_patch['champion'] == champ]
            if not r.empty:
                vec[idx] = float(r[pick_col].values[0]) + float(r[ban_col].values[0])
    meta_vectors[patch] = torch.tensor(vec, dtype=torch.float)

# Helper: average win_rate comfort for a list of players
def compute_comfort(players):
    vec = np.zeros(VOCAB_SIZE, dtype=float)
    for p in players:
        df_p = player_data.get(p)
        if df_p is None: continue
        for _, r in df_p.iterrows():
            idx = champ2idx.get(r['champion'])
            if idx:
                vec[idx] += float(r.get('win_rate', 0)) * 0.5
    if not players.empty:
        vec /= len(players)
    return torch.tensor(vec, dtype=torch.float)

# Dataset now yields sequence, meta, comfort, target
class DraftDataset(Dataset):
    def __init__(self, df):
        self.samples = []
        for _, row in df.iterrows():
            seq = encode_sequence(row)
            meta = meta_vectors.get(row.patch, torch.zeros(VOCAB_SIZE))
            # using blue team comfort; adjust as needed
            comfort = compute_comfort(row.blue_players)
            for t in range(1, len(seq)):
                inp = seq[:t] + [PAD_IDX] * (DRAFT_LENGTH - t)
                tgt = seq[t]
                self.samples.append((
                    torch.tensor(inp, dtype=torch.long),
                    meta,
                    comfort,
                    torch.tensor(tgt, dtype=torch.long)
                ))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# MLP model accepts meta and comfort
class MLPDraftModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.fc1 = nn.Linear(DRAFT_LENGTH * embedding_dim, hidden_dim)
        self.meta_fc = nn.Linear(VOCAB_SIZE, hidden_dim)
        self.comfort_fc = nn.Linear(VOCAB_SIZE, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, meta, comfort):
        emb = self.embedding(x)
        flat = emb.view(emb.size(0), -1)
        h = self.fc1(flat) + self.meta_fc(meta) + self.comfort_fc(comfort)
        h = F.relu(h)
        h = self.dropout(h)
        return self.fc2(h)
    def predict_next(self, seq, meta, comfort):
        self.eval()
        with torch.no_grad():
            inp = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
            m = meta.unsqueeze(0)
            c = comfort.unsqueeze(0)
            logits = self(inp, m, c)
            return idx2champ[int(torch.argmax(logits, dim=1).item())]

# RNN model accepts meta and comfort
class RNNDraftModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.meta_fc = nn.Linear(VOCAB_SIZE, hidden_dim)
        self.comfort_fc = nn.Linear(VOCAB_SIZE, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, meta, comfort):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        h = out[:, -1, :]
        h = h + self.meta_fc(meta) + self.comfort_fc(comfort)
        h = F.relu(h)
        return self.fc(h)
    def predict_next(self, seq, meta, comfort):
        self.eval()
        with torch.no_grad():
            inp = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
            m = meta.unsqueeze(0)
            c = comfort.unsqueeze(0)
            logits = self(inp, m, c)
            return idx2champ[int(torch.argmax(logits, dim=1).item())]

# Update training to unpack features
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, meta, comfort, y in train_loader:
            optimizer.zero_grad()
            logits = model(x, meta, comfort)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_acc, val_f1 = evaluate_model(model, val_loader)
        print(f"Epoch {epoch}/{epochs} - loss: {total_loss/len(train_loader):.4f}  val_acc: {val_acc:.4f}  val_f1: {val_f1:.4f}")

def evaluate_model(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, meta, comfort, y in loader:
            logits = model(x, meta, comfort)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            trues += y.cpu().numpy().tolist()
            preds += pred
    return accuracy_score(trues, preds), f1_score(trues, preds, average='macro')

# Example usage
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(draft_df, test_size=0.2, random_state=42)
    train_ds = DraftDataset(train_df)
    val_ds = DraftDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    mlp = MLPDraftModel()
    rnn = RNNDraftModel()
    print("Training MLP model...")
    train_model(mlp, train_loader, val_loader, epochs=10)
    print("Training RNN model...")
    train_model(rnn, train_loader, val_loader, epochs=10)
    torch.save(mlp.state_dict(), 'mlp_model.pt')
    torch.save(rnn.state_dict(), 'rnn_model.pt')
