import pandas as pd
import torch
import argparse
import json

# Load and normalize champion->roles mapping
with open('champion_roles.json') as f:
    raw_roles = json.load(f)
def _normalize(name: str) -> str:
    return name.lower().replace("'", "").replace(".", "").replace(" ", "")
NORM_ROLES = { _normalize(champ): roles for champ, roles in raw_roles.items() }
def get_roles(champ: str):
    return NORM_ROLES.get(_normalize(champ), [])

# Define roster order roles for dynamic mapping
ROLES_ORDER = ['Top','Jungle','Mid','ADC','Support']

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
def simulate_full(model, patch, blue_players, red_players, extra_bans=None):
    # Initialize empty draft state
    state = {
        'patch': patch,
        'blue_players': blue_players,
        'red_players': red_players,
        'blue_fs_bans': [], 'red_fs_bans': [],
        'blue_picks': [], 'blue_picks_players': [],
        'red_picks': [], 'red_picks_players': [],
        'blue_ss_bans': [], 'red_ss_bans': []
    }
    # enforce uniqueness and dynamic pick ordering
    used_idxs = set()
    # classification: map players to roles
    roles_map_blue = dict(zip(blue_players, ROLES_ORDER))
    roles_map_red = dict(zip(red_players, ROLES_ORDER))
    # apply fearless bans from prior games
    if extra_bans:
        for ch in extra_bans:
            idx = champ2idx.get(ch)
            if idx:
                used_idxs.add(idx)
    # remaining players per side for dynamic picks
    rem_blue = list(blue_players)
    rem_red = list(red_players)
    for event in EVENT_ORDER:
        row = pd.Series(state)
        # bag-of-picks feature: champs already picked
        meta = meta_vectors[patch]
        bag = torch.zeros_like(meta)
        for ch in state['blue_picks'] + state['red_picks']:
            idx = champ2idx.get(ch)
            if idx:
                bag[idx] = 1.0
        picks_tensor = bag.unsqueeze(0)
        # input sequence
        seq = encode_sequence(row) + [PAD_IDX] * (DRAFT_LENGTH - len(encode_sequence(row)))
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
                logits = model.forward_logits(inp_tensor, m, c_vec.unsqueeze(0), picks_tensor)
                # mask used indices
                for ui in used_idxs | {PAD_IDX}:
                    logits[0, ui] = float('-inf')
                # mask out champions not matching player's role
                player_role = roles_map_blue[player] if side == 'blue' else roles_map_red[player]
                for j in range(logits.size(1)):
                    champ_j = idx2champ.get(j)
                    if champ_j is None or player_role not in get_roles(champ_j):
                        logits[0, j] = float('-inf')
                pred_i = torch.argmax(logits, dim=1).item()
                score_i = logits[0, pred_i].item()
                if score_i > best_score:
                    best_score, best_pred_idx, best_player_idx = score_i, pred_i, i
            pred_idx = best_pred_idx
            pick = idx2champ[pred_idx]
            used_idxs.add(pred_idx)
            # record and remove selected player
            selected_player = rem_blue[best_player_idx] if side == 'blue' else rem_red[best_player_idx]
            if side == 'blue': del rem_blue[best_player_idx]
            else: del rem_red[best_player_idx]
        else:
            # bans: no comfort
            logits = model.forward_logits(inp_tensor, m, torch.zeros_like(meta).unsqueeze(0), picks_tensor)
            for ui in used_idxs | {PAD_IDX}:
                logits[0, ui] = float('-inf')
            pred_idx = torch.argmax(logits, dim=1).item()
            pick = idx2champ[pred_idx]
            used_idxs.add(pred_idx)
        state[event].append(pick)
        # log intended player for picks
        if 'picks' in event:
            state[event + '_players'].append(selected_player)
    return state

def predict_winner(state):
    # sum player comfort scores for picked champs
    b_score = sum(
        compute_comfort([player])[champ2idx.get(champ, PAD_IDX)].item()
        for player, champ in zip(state['blue_picks_players'], state['blue_picks'])
        if champ in champ2idx
    )
    r_score = sum(
        compute_comfort([player])[champ2idx.get(champ, PAD_IDX)].item()
        for player, champ in zip(state['red_picks_players'], state['red_picks'])
        if champ in champ2idx
    )
    # incorporate meta vector based on patch
    meta = meta_vectors[state['patch']]
    b_meta = sum(
        meta[champ2idx.get(champ, PAD_IDX)].item()
        for champ in state['blue_picks']
        if champ in champ2idx
    )
    r_meta = sum(
        meta[champ2idx.get(champ, PAD_IDX)].item()
        for champ in state['red_picks']
        if champ in champ2idx
    )
    # total score combining comfort and meta influence
    b_total = b_score + b_meta
    r_total = r_score + r_meta
    return 'blue' if b_total >= r_total else 'red'

def simulate_series(model, patch, blue_players, red_players, best_of=3):
    series = []
    extra_bans = []
    wins = {'blue': 0, 'red': 0}
    games_needed = best_of // 2 + 1
    for i in range(best_of):
        # swap sides each game
        if i % 2 == 1:
            b, r = red_players, blue_players
        else:
            b, r = blue_players, red_players
        state = simulate_full(model, patch, b, r, extra_bans)
        state['game_number'] = i + 1
        winner = predict_winner(state)
        state['winner'] = winner
        wins[winner] += 1
        series.append(state)
        # add all played champs to future bans
        for ev in EVENT_ORDER:
            for c in state[ev]:
                if c not in extra_bans:
                    extra_bans.append(c)
        if wins['blue'] == games_needed or wins['red'] == games_needed:
            break
    return series, wins

def run(best_of=3):
    patch = 13.5
    blue_players = ['Zeus','Peanut','Faker','Deft','Beryl']
    red_players =  ['Kiin','Oner','Knight','Ruler','Keria']
    mlp_series, mlp_wins = simulate_series(mlp, patch, blue_players, red_players, best_of)
    rnn_series, rnn_wins = simulate_series(rnn, patch, blue_players, red_players, best_of)
    print(f"=== MLP Best-of-{best_of} Series ===")
    for game in mlp_series:
        roles_map_blue = dict(zip(game['blue_players'], ROLES_ORDER))
        roles_map_red = dict(zip(game['red_players'], ROLES_ORDER))
        print(f"\nGame {game['game_number']} (Winner: {game['winner']})")
        print("  Phase 1 Bans  – Blue:", game['blue_fs_bans'], "Red:", game['red_fs_bans'])
        # annotate Phase 1 picks with roles and check off-role
        blue_phase1 = [(p, c, get_roles(c), roles_map_blue.get(p) in get_roles(c))
                       for p, c in zip(game['blue_picks_players'][:3], game['blue_picks'][:3])]
        red_phase1 = [(p, c, get_roles(c), roles_map_red.get(p) in get_roles(c))
                      for p, c in zip(game['red_picks_players'][:3], game['red_picks'][:3])]
        print("  Phase 1 Picks – Blue:", blue_phase1)
        print("                  Red:", red_phase1)
        print("  Phase 2 Bans  – Blue:", game['blue_ss_bans'], "Red:", game['red_ss_bans'])
        # annotate Phase 2 picks with roles and check off-role
        blue_phase2 = [(p, c, get_roles(c), roles_map_blue.get(p) in get_roles(c))
                       for p, c in zip(game['blue_picks_players'][3:], game['blue_picks'][3:])]
        red_phase2 = [(p, c, get_roles(c), roles_map_red.get(p) in get_roles(c))
                      for p, c in zip(game['red_picks_players'][3:], game['red_picks'][3:])]
        print("  Phase 2 Picks – Blue:", blue_phase2)
        print("                  Red:", red_phase2)
    print(f"\nSeries result: Blue {mlp_wins['blue']} – {mlp_wins['red']} Red\n")

    print(f"=== RNN Best-of-{best_of} Series ===")
    for game in rnn_series:
        roles_map_blue = dict(zip(game['blue_players'], ROLES_ORDER))
        roles_map_red = dict(zip(game['red_players'], ROLES_ORDER))
        print(f"\nGame {game['game_number']} (Winner: {game['winner']})")
        print("  Phase 1 Bans  – Blue:", game['blue_fs_bans'], "Red:", game['red_fs_bans'])
        # annotate Phase 1 picks with roles and check off-role
        blue_phase1 = [(p, c, get_roles(c), roles_map_blue.get(p) in get_roles(c))
                       for p, c in zip(game['blue_picks_players'][:3], game['blue_picks'][:3])]
        red_phase1 = [(p, c, get_roles(c), roles_map_red.get(p) in get_roles(c))
                      for p, c in zip(game['red_picks_players'][:3], game['red_picks'][:3])]
        print("  Phase 1 Picks – Blue:", blue_phase1)
        print("                  Red:", red_phase1)
        print("  Phase 2 Bans  – Blue:", game['blue_ss_bans'], "Red:", game['red_ss_bans'])
        # annotate Phase 2 picks with roles and check off-role
        blue_phase2 = [(p, c, get_roles(c), roles_map_blue.get(p) in get_roles(c))
                       for p, c in zip(game['blue_picks_players'][3:], game['blue_picks'][3:])]
        red_phase2 = [(p, c, get_roles(c), roles_map_red.get(p) in get_roles(c))
                      for p, c in zip(game['red_picks_players'][3:], game['red_picks'][3:])]
        print("  Phase 2 Picks – Blue:", blue_phase2)
        print("                  Red:", red_phase2)
    print(f"Series result: Blue {rnn_wins['blue']} – {rnn_wins['red']} Red")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a best-of-series draft")
    parser.add_argument('-b', '--best-of', type=int, default=3, choices=[1,3,5],
                        help='Series length (best-of): 1, 3, or 5')
    args = parser.parse_args()
    run(best_of=args.best_of)
