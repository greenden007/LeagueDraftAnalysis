import nn_preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

NUM_CHAMPIONS = 170
NUM_ROLES = 5
DRAFT_LENGTH = 20

INPUT_SIZE = NUM_CHAMPIONS * DRAFT_LENGTH
ROLE_NAMES = ["top", "jungle", "mid", "bot", "support"]

draft_df = nn_preprocessing.load_full_draft_csv() # all drafts dataframe
player_data = nn_preprocessing.load_player_data() # player dict[str, pd.DataFrame], sorted by player, relationship with champs
pickban_df = nn_preprocessing.load_pickban_data() # pickban dict[str, pd.DataFrame], sorted by patch
deft_data = player_data["Deft"]
print(deft_data)

def encode_draft_state(picks_bans, player_roles, patch_vector=None):
    """
    Encodes the current draft state into a flat vector.
    picks_bans: list of length 20, filled with champion IDs or -1 for not chosen
    player_roles: dict with role names as keys and player data vectors as values
    patch_vector: optional one-hot or stat vector for current patch
    """
    one_hot = torch.zeros(NUM_CHAMPIONS * DRAFT_LENGTH)
    for i, champ in enumerate(picks_bans):
        if champ != -1:
            one_hot[i * NUM_CHAMPIONS + champ] = 1.0

    player_features = torch.cat([player_roles[role] for role in ROLE_NAMES])
    if patch_vector is not None:
        return torch.cat([one_hot, player_features, patch_vector])
    else:
        return torch.cat([one_hot, player_features])

def build_training_data_from_drafts(draft_df, player_data, pickban_df):
    """
    Convert drafts into supervised learning format: (state, next_action)
    """
    X, y = [], []
    for _, row in draft_df.iterrows():
        picks_bans = [-1] * DRAFT_LENGTH
        # If player_data[row[role]]['features'] is already a tensor, use .float(); otherwise, use torch.tensor(...)
        player_roles = {role: torch.tensor(player_data[row[role]]['features'].values, dtype=torch.float32)
                        if not torch.is_tensor(player_data[row[role]]['features'].values) else player_data[row[role]]['features'].values.float()
                        for role in ROLE_NAMES}
        # Same for patch_vector
        patch_vector_raw = pickban_df[row['patch']]['vector'].values
        patch_vector = torch.tensor(patch_vector_raw, dtype=torch.float32) if not torch.is_tensor(patch_vector_raw) else patch_vector_raw.float()

        for i in range(DRAFT_LENGTH):
            champ = row[f'step_{i}']
            x_vec = encode_draft_state(picks_bans, player_roles, patch_vector)
            X.append(x_vec)
            y.append(champ)
            picks_bans[i] = champ

    return torch.stack(X), torch.tensor(y)

def simulate_draft(model, player_roles, patch_vector, human_side="blue", series_picked_champions=None):
    """
    Simulate a full League of Legends draft (with fearless draft support) using the official pick/ban order.
    - Ban phase 1: Blue ban, Red ban (repeat 3 times)
    - Pick phase 1: Blue pick, Red 2 picks, Blue 2 picks, Red 1 pick
    - Ban phase 2: Blue ban, Red ban (repeat 2 times)
    - Pick phase 2: Red 1 pick, Blue 2 picks, Red 1 pick

    Args:
        model: The draft prediction model (MLP or RNN).
        player_roles: dict mapping role names to player feature tensors.
        patch_vector: tensor for current patch features.
        human_side: 'blue' or 'red'.
        series_picked_champions: set of champion IDs picked in previous games (for fearless draft). If None, initializes as empty set.
    Returns:
        picks_bans: list of picks/bans in draft order (length 20).
    """
    model.eval()
    if series_picked_champions is None:
        series_picked_champions = set()
    picks_bans = [-1] * DRAFT_LENGTH
    step = 0
    # Draft order definition (step, phase, side, action)
    draft_sequence = [
        # Ban Phase 1
        ("ban", "blue"), ("ban", "red"),
        ("ban", "blue"), ("ban", "red"),
        ("ban", "blue"), ("ban", "red"),
        # Pick Phase 1
        ("pick", "blue"), ("pick", "red"), ("pick", "red"),
        ("pick", "blue"), ("pick", "blue"), ("pick", "red"),
        # Ban Phase 2
        ("ban", "blue"), ("ban", "red"),
        ("ban", "blue"), ("ban", "red"),
        # Pick Phase 2
        ("pick", "red"), ("pick", "blue"), ("pick", "blue"), ("pick", "red")
    ]
    # Track picks and bans separately for clarity
    picks = {"blue": [], "red": []}
    bans = {"blue": [], "red": []}
    for phase, side in draft_sequence:
        # Compute available champions (not picked/banned this game or in series)
        unavailable = set(picks["blue"]) | set(picks["red"]) | set(bans["blue"]) | set(bans["red"]) | set(series_picked_champions)
        available = list(set(range(NUM_CHAMPIONS)) - unavailable)
        x_vec = encode_draft_state(picks_bans, player_roles, patch_vector).unsqueeze(0)
        pred = torch.argmax(model(x_vec)).item()
        is_human = (side == human_side)
        action_str = "Pick" if phase == "pick" else "Ban"
        if is_human:
            print(f"Draft step {step+1}: Your turn to {action_str.lower()} for {side}. Available IDs: {sorted(available)}")
            while True:
                try:
                    user_input = int(input(f"Enter champion ID to {action_str.lower()}: "))
                    if user_input in available:
                        champ_id = user_input
                        break
                    else:
                        print("Invalid or already picked/banned champion ID (or not allowed by fearless draft). Try again.")
                except ValueError:
                    print("Please enter a valid integer.")
        else:
            champ_id = pred
            # AI must also respect fearless draft and draft state
            if champ_id not in available:
                # Choose the top available champion (fallback)
                champ_id = available[0]
            print(f"Draft step {step+1}: AI {action_str.lower()}s {champ_id} for {side}")
        # Record action
        picks_bans[step] = champ_id
        if phase == "pick":
            picks[side].append(champ_id)
            series_picked_champions.add(champ_id)  # Fearless draft: add to series-level set
        else:
            bans[side].append(champ_id)
        step += 1
    print(f"\nDraft complete! Blue picks: {picks['blue']}, Red picks: {picks['red']}")
    print(f"Blue bans: {bans['blue']}, Red bans: {bans['red']}")
    return picks_bans


class MLPDraftModel(nn.Module):
    """
    Multi-Layer Perceptron model for draft prediction.
    """
    def __init__(self, input_size: int, hidden_sizes=[512, 256], output_size: int = NUM_CHAMPIONS) -> None:
        super(MLPDraftModel, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

class RNNDraftModel(nn.Module):
    """
    GRU-based RNN model for draft prediction.
    """
    def __init__(self, input_size: int = NUM_CHAMPIONS, hidden_size: int = 256, num_layers: int = 2, output_size: int = NUM_CHAMPIONS) -> None:
        super(RNNDraftModel, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects x of shape (batch, seq_len, input_size)."""
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

def build_rnn_training_data_from_drafts(draft_df, player_data, pickban_df, seq_len=5):
    """
    Builds (X_seq, y_seq) for RNNs, where each X_seq is a sequence of draft states,
    and y_seq is the next champion for each step in the sequence.
    """
    X_seqs, y_seqs = [], []
    for _, row in draft_df.iterrows():
        picks_bans = [-1] * DRAFT_LENGTH
        player_roles = {role: torch.tensor(player_data[row[role]]['features'].values, dtype=torch.float32)
                        if not torch.is_tensor(player_data[row[role]]['features'].values) else player_data[row[role]]['features'].values.float()
                        for role in ROLE_NAMES}
        patch_vector_raw = pickban_df[row['patch']]['vector'].values
        patch_vector = torch.tensor(patch_vector_raw, dtype=torch.float32) if not torch.is_tensor(patch_vector_raw) else patch_vector_raw.float()
        draft_states = []
        for i in range(DRAFT_LENGTH):
            x_vec = encode_draft_state(picks_bans, player_roles, patch_vector)
            draft_states.append(x_vec)
            picks_bans[i] = row[f'step_{i}']
        # Create rolling sequences
        for i in range(DRAFT_LENGTH - seq_len):
            X_seqs.append(torch.stack(draft_states[i:i+seq_len]))
            y_seqs.append(row[f'step_{i+seq_len}'])
    return torch.stack(X_seqs), torch.tensor(y_seqs)

def train_model(model, train_X, train_y, val_X=None, val_y=None, epochs=10, batch_size=64, lr=1e-3, is_rnn=False, device='cpu'):
    """
    Generic PyTorch training loop for both MLP and RNN models.
    train_X, train_y: tensors (MLP: [N, input_size], RNN: [N, seq_len, input_size])
    val_X, val_y: optional validation tensors
    """
    from torch.utils.data import DataLoader, TensorDataset
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_ds = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    if val_X is not None and val_y is not None:
        val_ds = TensorDataset(val_X, val_y)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
    else:
        val_loader = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if is_rnn:
                X_batch = X_batch.float()  # (batch, seq_len, input_size)
            else:
                X_batch = X_batch.float()  # (batch, input_size)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Train loss = {total_loss/len(train_loader):.4f}")
        # Optionally: evaluate on val_loader using eval_model
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    if is_rnn:
                        X_val = X_val.float()
                    else:
                        X_val = X_val.float()
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    val_loss += loss.item()
            print(f"Epoch {epoch+1}: Val loss = {val_loss/len(val_loader):.4f}")

def eval_model(model, data_loader, device: str = "cpu", topk=1, draft_sequence=None) -> tuple:
    """
    Evaluate the model on a dataset.
    Returns (accuracy, f1_score).
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_step_types = []
    all_topk_hits = []
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                inputs, labels, step_types = batch["inputs"], batch["labels"], batch["step_types"]
            else:
                inputs, labels = batch
                step_types = None
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if step_types is not None:
                all_step_types.extend(step_types)
            if topk > 1:
                topk_preds = torch.topk(outputs, k=topk, dim=1).indices.cpu().numpy()
                hits = [label in topk_pred for label, topk_pred in zip(labels.cpu().numpy(), topk_preds)]
                all_topk_hits.extend(hits)
    metrics = {}
    metrics["accuracy"] = accuracy_score(all_labels, all_preds)
    metrics["f1"] = f1_score(all_labels, all_preds, average='weighted')
    if topk > 1:
        metrics["topk_hits"] = sum(all_topk_hits) / len(all_topk_hits)
    if all_step_types:
        phase_metrics = defaultdict(lambda: {"labels": [], "preds": []})
        for label, pred, step_type in zip(all_labels, all_preds, all_step_types):
            phase_metrics[step_type]["labels"].append(label)
            phase_metrics[step_type]["preds"].append(pred)
        for phase in phase_metrics:
            metrics[f"{phase}_accuracy"] = accuracy_score(phase_metrics[phase]["labels"], phase_metrics[phase]["preds"])
            metrics[f"{phase}_f1"] = f1_score(phase_metrics[phase]["labels"], phase_metrics[phase]["preds"], average='weighted')
    return metrics

# X, y = build_training_data_from_drafts(draft_df, player_data, pickban_df)
# train_model(MLPDraftModel(INPUT_SIZE), X, y, epochs=10)

# X_seq, y_seq = build_rnn_training_data_from_drafts(draft_df, player_data, pickban_df, seq_len=5)
# train_model(RNNDraftModel(input_size=X_seq.shape[2]), X_seq, y_seq, epochs=10, is_rnn=True)


