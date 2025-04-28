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
    if (len(row.blue_fs_bans) < 3 or len(row.red_fs_bans) < 3):
        row.blue_fs_bans = row.blue_fs_bans + [PAD_IDX] * (3 - len(row.blue_fs_bans))
        row.red_fs_bans = row.red_fs_bans + [PAD_IDX] * (3 - len(row.red_fs_bans))
    seq = []
    
    # phase1 bans
    seq.append(champ2idx.get(row.blue_fs_bans[0], PAD_IDX))
    seq.append(champ2idx.get(row.red_fs_bans[0], PAD_IDX))
    seq.append(champ2idx.get(row.blue_fs_bans[1], PAD_IDX))
    seq.append(champ2idx.get(row.red_fs_bans[1], PAD_IDX))
    seq.append(champ2idx.get(row.blue_fs_bans[2], PAD_IDX))
    seq.append(champ2idx.get(row.red_fs_bans[2], PAD_IDX))

    # prepare sorted picks, ignore game pick order for encoding
    b_sorted = sorted(row.blue_picks)
    if len(b_sorted) < 5:
        b_sorted += [None] * (5 - len(b_sorted))
    r_sorted = sorted(row.red_picks)
    if len(r_sorted) < 5:
        r_sorted += [None] * (5 - len(r_sorted))
    # phase1 sorted picks: first 3 Blues then 3 Reds
    seq.extend(champ2idx.get(p, PAD_IDX) for p in b_sorted[:3])
    seq.extend(champ2idx.get(p, PAD_IDX) for p in r_sorted[:3])

    if (len(row.blue_ss_bans) < 2 or len(row.red_ss_bans) < 2):
        row.blue_ss_bans = row.blue_ss_bans + [PAD_IDX] * (2 - len(row.blue_ss_bans))
        row.red_ss_bans = row.red_ss_bans + [PAD_IDX] * (2 - len(row.red_ss_bans))
    
    # phase2 bans
    seq.append(champ2idx.get(row.red_ss_bans[0], PAD_IDX))
    seq.append(champ2idx.get(row.blue_ss_bans[0], PAD_IDX))
    seq.append(champ2idx.get(row.red_ss_bans[1], PAD_IDX))
    seq.append(champ2idx.get(row.blue_ss_bans[1], PAD_IDX))

    # phase2 sorted picks: remaining 2 Blues then 2 Reds
    seq.extend(champ2idx.get(p, PAD_IDX) for p in b_sorted[3:5])
    seq.extend(champ2idx.get(p, PAD_IDX) for p in r_sorted[3:5])
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
                vec[idx] += float(r.get('win_rate', 0))
    if len(players) > 0:
        vec /= len(players)
    return torch.tensor(vec, dtype=torch.float)

# Dataset now yields sequence, bag, meta, comfort, target
class DraftDataset(Dataset):
    def __init__(self, df):
        self.samples = []
        for _, row in df.iterrows():
            seq = encode_sequence(row)
            # bag-of-picks feature: which champs have been picked so far
            bag = torch.zeros(VOCAB_SIZE)
            for p in row.blue_picks + row.red_picks:
                idx = champ2idx.get(p)
                if idx:
                    bag[idx] = 1.0
            meta = meta_vectors.get(row.patch, torch.zeros(VOCAB_SIZE))
            comfort = compute_comfort(row.blue_players)
            for t in range(1, len(seq)):
                inp = seq[:t] + [PAD_IDX] * (DRAFT_LENGTH - t)
                tgt = seq[t]
                self.samples.append((
                    torch.tensor(inp, dtype=torch.long),
                    bag.clone(),
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
        self.picks_fc = nn.Linear(VOCAB_SIZE, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, meta, comfort, picks):
        emb = self.embedding(x)
        flat = emb.view(emb.size(0), -1)
        h = self.fc1(flat) + self.meta_fc(meta) + self.comfort_fc(comfort) + self.picks_fc(picks)
        h = F.relu(h)
        h = self.dropout(h)
        return self.fc2(h)
    def forward_logits(self, x, meta, comfort, picks):
        """
        Returns raw logits tensor for all champion indices.
        """
        return self(x, meta, comfort, picks)
    def predict_next(self, seq, meta, comfort, picks):
        self.eval()
        with torch.no_grad():
            inp = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
            m = meta.unsqueeze(0)
            c = comfort.unsqueeze(0)
            p = picks.unsqueeze(0)
            logits = self(inp, m, c, p)
            logits[:, 0] = float('-inf')            # ban PAD
            pred = torch.argmax(logits, dim=1).item()
            return idx2champ[pred]

# RNN model accepts meta and comfort
class RNNDraftModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.meta_fc = nn.Linear(VOCAB_SIZE, hidden_dim)
        self.comfort_fc = nn.Linear(VOCAB_SIZE, hidden_dim)
        self.picks_fc = nn.Linear(VOCAB_SIZE, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, meta, comfort, picks):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        h = out[:, -1, :]
        h = h + self.meta_fc(meta) + self.comfort_fc(comfort) + self.picks_fc(picks)
        h = F.relu(h)
        return self.fc(h)
    def forward_logits(self, x, meta, comfort, picks):
        """
        Returns raw logits tensor for all champion indices.
        """
        return self(x, meta, comfort, picks)
    def predict_next(self, seq, meta, comfort, picks):
        self.eval()
        with torch.no_grad():
            inp = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
            m = meta.unsqueeze(0)
            c = comfort.unsqueeze(0)
            p = picks.unsqueeze(0)
            logits = self(inp, m, c, p)
            logits[:, 0] = float('-inf')            # ban PAD
            pred = torch.argmax(logits, dim=1).item()
            return idx2champ[pred]

# Update training to unpack features
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, bag, meta, comfort, y in train_loader:
            optimizer.zero_grad()
            logits = model(x, meta, comfort, bag)
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
        for x, bag, meta, comfort, y in loader:
            logits = model(x, meta, comfort, bag)
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
    train_model(mlp, train_loader, val_loader, epochs=12)
    print("Training RNN model...")
    train_model(rnn, train_loader, val_loader, epochs=12)
    torch.save(mlp.state_dict(), 'mlp_model.pt')
    torch.save(rnn.state_dict(), 'rnn_model.pt')
