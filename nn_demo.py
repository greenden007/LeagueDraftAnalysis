import pandas as pd
import torch
from nn_learn import (
    PAD_IDX, DRAFT_LENGTH, MLPDraftModel, RNNDraftModel, encode_sequence, champ2idx, meta_vectors, compute_comfort, idx2champ
)
# Initialize models
vocab_size = len(champ2idx) + 1
mlp = MLPDraftModel(vocab_size=vocab_size)
rnn = RNNDraftModel(vocab_size=vocab_size)
mlp.load_state_dict(torch.load("mlp_model.pt"))
rnn.load_state_dict(torch.load("rnn_model.pt"))
mlp.eval()
rnn.eval()

# Define event order for fearless draft (20 steps)
EVENT_ORDER = [
    'blue_fs_bans','red_fs_bans','blue_fs_bans','red_fs_bans','blue_fs_bans','red_fs_bans',  # Phase1 bans
    'blue_picks','red_picks','red_picks','blue_picks','blue_picks','red_picks',           # Phase1 picks
    'blue_ss_bans','red_ss_bans','blue_ss_bans','red_ss_bans',                            # Phase2 bans
    'red_picks','blue_picks','blue_picks','red_picks'                                     # Phase2 picks
]

# Simulation function
def simulate_full(model, patch, blue_players, red_players):
    # Initialize empty draft state
    state = {
        'patch': patch,
        'blue_players': blue_players,
        'red_players': red_players,
        'blue_fs_bans': [], 'red_fs_bans': [],
        'blue_picks': [], 'red_picks': [],
        'blue_ss_bans': [], 'red_ss_bans': []
    }
    # enforce uniqueness and dynamic pick ordering
    used_idxs = set()
    # remaining players per side for dynamic picks
    rem_blue = list(blue_players)
    rem_red = list(red_players)
    for event in EVENT_ORDER:
        row = pd.Series(state)
        seq = encode_sequence(row) + [PAD_IDX] * (DRAFT_LENGTH - len(encode_sequence(row)))
        meta = meta_vectors[patch]
        inp_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        m = meta.unsqueeze(0)
        # dynamic selection for picks vs static for bans
        if 'picks' in event:
            side = 'blue' if event.startswith('blue') else 'red'
            best_score = float('-inf')
            best_pred_idx = None
            best_player_idx = None
            pool = rem_blue if side == 'blue' else rem_red
            for i, player in enumerate(pool):
                c_vec = compute_comfort([player])
                logits = model.forward_logits(inp_tensor, m, c_vec.unsqueeze(0))
                for ui in used_idxs | {PAD_IDX}:
                    logits[0, ui] = float('-inf')
                pred_i = torch.argmax(logits, dim=1).item()
                score_i = logits[0, pred_i].item()
                if score_i > best_score:
                    best_score, best_pred_idx, best_player_idx = score_i, pred_i, i
            pred_idx = best_pred_idx
            pick = idx2champ[pred_idx]
            used_idxs.add(pred_idx)
            # remove selected player
            if side == 'blue': del rem_blue[best_player_idx]
            else: del rem_red[best_player_idx]
        else:
            # bans: no comfort
            logits = model.forward_logits(inp_tensor, m, torch.zeros_like(meta).unsqueeze(0))
            for ui in used_idxs | {PAD_IDX}:
                logits[0, ui] = float('-inf')
            pred_idx = torch.argmax(logits, dim=1).item()
            pick = idx2champ[pred_idx]
            used_idxs.add(pred_idx)
        state[event].append(pick)
    return state

# Run full draft simulation
def run():
    patch = 14.10
    blue_players = ['Zeus','Peanut','Faker','Deft','Beryl']
    red_players =  ['Doran','Oner','Chovy','Viper','Busio']
    mlp_state = simulate_full(mlp, patch, blue_players, red_players)
    rnn_state = simulate_full(rnn, patch, blue_players, red_players)
    print("=== MLP Full Draft Simulation ===")
    for phase, champs in mlp_state.items():
        if isinstance(champs, list): print(f"{phase}: {champs}")
    print("\n=== RNN Full Draft Simulation ===")
    for phase, champs in rnn_state.items():
        if isinstance(champs, list): print(f"{phase}: {champs}")

if __name__ == "__main__":
    run()
