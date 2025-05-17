import pandas as pd
import torch
import argparse
import json
from collections import defaultdict
from heuristic_agent import DraftAgent

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
        # Heuristic agent path
        if isinstance(model, DraftAgent):
            # build current bans and picks for heuristic
            bans = state['blue_fs_bans'] + state['red_fs_bans'] + state['blue_ss_bans'] + state['red_ss_bans']
            picks = [(p, roles_map_blue[player], 'blue') for p, player in zip(state['blue_picks'], state['blue_picks_players'])]
            picks += [(p, roles_map_red[player], 'red') for p, player in zip(state['red_picks'], state['red_picks_players'])]
            phase = 'Pick' if 'picks' in event else 'Ban'
            model.update_draft_state(bans=bans, picks=picks, phase=phase)
            suggestion = model.get_suggestion()
            if 'bans' in event:
                bans_list = suggestion.get('recommended_bans', [])
                if bans_list:
                    pick = bans_list[0]
                else:
                    # fallback to first unused champion
                    pick = next((c for c, idx in champ2idx.items() if idx not in used_idxs), None)
                selected_player = None
            else:
                recs = suggestion.get('recommended_picks', {})
                pick = None; selected_player = None
                # choose first role recommendation matching available players
                for role in ROLES_ORDER:
                    if recs.get(role):
                        candidate = recs[role][0]
                        pool = rem_blue if event.startswith('blue') else rem_red
                        roles_map = roles_map_blue if event.startswith('blue') else roles_map_red
                        # find player for this role
                        for pl in list(pool):
                            if roles_map[pl] == role:
                                pick = candidate
                                selected_player = pl
                                pool.remove(pl)
                                break
                        if pick:
                            break
                # fallback to first unused champ if no recommendation
                if not pick:
                    pick = next((c for c, idx in champ2idx.items() if idx not in used_idxs), None)
                    selected_player = None
            idx = champ2idx.get(pick)
            if idx is not None:
                used_idxs.add(idx)
            state[event].append(pick)
            if selected_player:
                state[event + '_players'].append(selected_player)
            continue
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
        encoded_row = encode_sequence(row)
        seq = encoded_row + [PAD_IDX] * (DRAFT_LENGTH - len(encoded_row))
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

def simulate_matchups(num_simulations=500, series_lengths=[1,3,5]):
    # Initialize models
    vocab_size = len(champ2idx) + 1
    mlp = MLPDraftModel(vocab_size=vocab_size)
    rnn = RNNDraftModel(vocab_size=vocab_size)
    mlp.load_state_dict(torch.load("mlp_model.pt"))
    rnn.load_state_dict(torch.load("rnn_model.pt"))
    mlp.eval()
    rnn.eval()

    # Initialize results storage
    results = {
        'MLP_vs_MLP': {
            'blue': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0},
            'red': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0}
        },
        'MLP_vs_RNN': {
            'blue': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0},
            'red': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0}
        },
        'RNN_vs_MLP': {
            'blue': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0},
            'red': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0}
        },
        'RNN_vs_RNN': {
            'blue': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0},
            'red': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0}
        },
        'MLP_vs_Heuristic': {
            'blue': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0},
            'red': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0}
        },
        'Heuristic_vs_MLP': {
            'blue': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0},
            'red': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0}
        },
        'RNN_vs_Heuristic': {
            'blue': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0},
            'red': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0}
        },
        'Heuristic_vs_RNN': {
            'blue': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0},
            'red': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0}
        },
        'Heuristic_vs_Heuristic': {
            'blue': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0},
            'red': {'picks': defaultdict(int), 'bans': defaultdict(int), 'wins': 0, 'game_wins': 0}
        },
        'series_results': {length: {
            'MLP_vs_MLP': {'blue': 0, 'red': 0},
            'MLP_vs_RNN': {'blue': 0, 'red': 0},
            'RNN_vs_MLP': {'blue': 0, 'red': 0},
            'RNN_vs_RNN': {'blue': 0, 'red': 0},
            'MLP_vs_Heuristic': {'blue': 0, 'red': 0},
            'Heuristic_vs_MLP': {'blue': 0, 'red': 0},
            'RNN_vs_Heuristic': {'blue': 0, 'red': 0},
            'Heuristic_vs_RNN': {'blue': 0, 'red': 0},
            'Heuristic_vs_Heuristic': {'blue': 0, 'red': 0}
        } for length in series_lengths},
        'draft_sequences': {
            'MLP_vs_MLP': [],
            'MLP_vs_RNN': [],
            'RNN_vs_MLP': [],
            'RNN_vs_RNN': [],
            'MLP_vs_Heuristic': [],
            'Heuristic_vs_MLP': [],
            'RNN_vs_Heuristic': [],
            'Heuristic_vs_RNN': [],
            'Heuristic_vs_Heuristic': []
        },
        # Initialize per-series-length game win breakdown
        'series_game_breakdown': {length: [] for length in series_lengths}
    }

    # Sample players for simulations
    blue_players = ['Zeus','Peanut','Faker','Deft','Beryl']
    red_players = ['Kiin','Oner','Knight','Ruler','Keria']

    # Run simulations
    for _ in range(num_simulations):
        # Run individual games
        all_matchups = [
            'MLP_vs_MLP', 'MLP_vs_RNN', 'RNN_vs_MLP', 'RNN_vs_RNN',
            'MLP_vs_Heuristic', 'Heuristic_vs_MLP', 'RNN_vs_Heuristic', 'Heuristic_vs_RNN', 'Heuristic_vs_Heuristic'
        ]
        for matchup in all_matchups:
            # Determine agent types
            blue_agent = None
            red_agent = None
            if matchup.startswith('MLP_vs_'):
                blue_agent = mlp
            elif matchup.startswith('RNN_vs_'):
                blue_agent = rnn
            elif matchup.startswith('Heuristic_vs_'):
                blue_agent = DraftAgent(side='blue', patch='25.5')
            if matchup.endswith('_vs_MLP'):
                red_agent = mlp
            elif matchup.endswith('_vs_RNN'):
                red_agent = rnn
            elif matchup.endswith('_vs_Heuristic'):
                red_agent = DraftAgent(side='red', patch='25.5')
            # Heuristic vs Heuristic
            if matchup == 'Heuristic_vs_Heuristic':
                blue_agent = DraftAgent(side='blue', patch='25.5')
                red_agent = DraftAgent(side='red', patch='25.5')
            # Fallback to mlp/rnn if not set
            if blue_agent is None:
                blue_agent = mlp
            if red_agent is None:
                red_agent = rnn
            # Run single game
            state = simulate_full(blue_agent, 13.10, blue_players, red_players)
            winner = predict_winner(state)
            results[matchup][winner]['wins'] += 1
            for champ in state['blue_picks']:
                results[matchup]['blue']['picks'][champ] += 1
            for champ in state['red_picks']:
                results[matchup]['red']['picks'][champ] += 1

        # Run series simulations
        for length in series_lengths:
            for matchup in all_matchups:
                # Determine agent types
                blue_agent = None
                red_agent = None
                if matchup.startswith('MLP_vs_'):
                    blue_agent = mlp
                elif matchup.startswith('RNN_vs_'):
                    blue_agent = rnn
                elif matchup.startswith('Heuristic_vs_'):
                    blue_agent = DraftAgent(side='blue', patch='25.5')
                if matchup.endswith('_vs_MLP'):
                    red_agent = mlp
                elif matchup.endswith('_vs_RNN'):
                    red_agent = rnn
                elif matchup.endswith('_vs_Heuristic'):
                    red_agent = DraftAgent(side='red', patch='25.5')
                if matchup == 'Heuristic_vs_Heuristic':
                    blue_agent = DraftAgent(side='blue', patch='25.5')
                    red_agent = DraftAgent(side='red', patch='25.5')
                if blue_agent is None:
                    blue_agent = mlp
                if red_agent is None:
                    red_agent = rnn
                # Run series with proper ban carryover and track individual games
                series_bans = []
                series_result = simulate_series(blue_agent, 13.10, blue_players, red_players, best_of=length)
                series_states = series_result[0]  # List of game states
                wins = series_result[1]  # Wins count
                
                # Track individual game results
                for game_state in series_states:
                    # Track picks and bans
                    # First track bans
                    for team, bans in [('blue', game_state['blue_fs_bans'] + game_state['blue_ss_bans']),
                                      ('red', game_state['red_fs_bans'] + game_state['red_ss_bans'])]:
                        for champ in bans:
                            results[matchup][team]['bans'][champ] += 1
                    
                    # Then track picks
                    for champ in game_state['blue_picks']:
                        results[matchup]['blue']['picks'][champ] += 1
                    for champ in game_state['red_picks']:
                        results[matchup]['red']['picks'][champ] += 1

                    # Track draft sequence
                    draft_seq = {
                        'blue': {
                            'fs_bans': game_state['blue_fs_bans'],
                            'ss_bans': game_state['blue_ss_bans'],
                            'picks': game_state['blue_picks']
                        },
                        'red': {
                            'fs_bans': game_state['red_fs_bans'],
                            'ss_bans': game_state['red_ss_bans'],
                            'picks': game_state['red_picks']
                        }
                    }
                    results['draft_sequences'][matchup].append(draft_seq)
                    
                    # Track game winner
                    game_winner = predict_winner(game_state)
                    results[matchup][game_winner]['game_wins'] += 1
                    
                    # Add bans for next game
                    for ev in EVENT_ORDER:
                        if 'picks' in ev:
                            for champ in game_state[ev]:
                                if champ not in series_bans:
                                    series_bans.append(champ)
                
                # Determine series winner
                series_winner = 'blue' if wins['blue'] > wins['red'] else 'red'
                results['series_results'][length][matchup][series_winner] += 1
                
                # Record this series' winner sequence as model names
                series_winners = []
                for gs in series_states:
                    side = predict_winner(gs)
                    model_name = matchup.split('_vs_')[0] if side == 'blue' else matchup.split('_vs_')[1]
                    if model_name == 'Heuristic':
                        model_name = 'HSTC'
                    series_winners.append(model_name)
                results['series_game_breakdown'][length].append(series_winners)

    # Count most common champion picks by model type
    model_picks = {
        'MLP': defaultdict(int),
        'RNN': defaultdict(int),
        'Heuristic': defaultdict(int)
    }
    for matchup, data in results.items():
        if matchup in ['series_results','draft_sequences','series_game_breakdown']:
            continue
        blue_model, red_model = matchup.split('_vs_')
        for champ, cnt in data['blue']['picks'].items():
            model_picks[blue_model][champ] += cnt
        for champ, cnt in data['red']['picks'].items():
            model_picks[red_model][champ] += cnt
    results['model_picks'] = { model: dict(picks) for model, picks in model_picks.items() }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a best-of-series draft")
    parser.add_argument('-b', '--best-of', type=int, default=3, choices=[1,3,5],
                        help='Series length (best-of): 1, 3, or 5')
    parser.add_argument('--matchups', action='store_true',
                        help='Run matchup simulations')
    parser.add_argument('--num-sims', type=int, default=1,
                        help='Number of simulations to run')
    args = parser.parse_args()

    if args.matchups:
        results = simulate_matchups(num_simulations=args.num_sims)
        
        # Write results to file
        with open('matchup_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print matchup results
        print("\nMatchup Simulation Results:\n")
        for matchup, data in results.items():
            if matchup == 'series_results':
                print("Series Results:")
                for length, length_data in data.items():
                    print(f"\nBest-of-{length}:")
                    for matchup_type, matchup_data in length_data.items():
                        print(f"  {matchup_type}:")
                        print(f"    Blue wins: {matchup_data['blue']}")
                        print(f"    Red wins: {matchup_data['red']}")
            elif matchup == 'series_game_breakdown' or matchup == 'draft_sequences':
                # skip breakdown keys
                continue
            else:
                blue_model, red_model = matchup.split('_vs_')
                print(f"{matchup}:")
                print(f"  {blue_model} wins: {data['blue']['wins']} (Game wins: {data['blue']['game_wins']})")
                print(f"  {red_model} wins: {data['red']['wins']} (Game wins: {data['red']['game_wins']})")
                print("  Top 5 picks for each team:")
                for team, team_data in data.items():
                    print(f"    {team} team:")
                    
                    # Print top picks
                    print("      Top picks:")
                    top_picks = sorted(team_data['picks'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for champ, count in top_picks:
                        print(f"        {champ}: {count} picks")
                    
                    # Print top bans
                    print("      Top bans:")
                    top_bans = sorted(team_data['bans'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for champ, count in top_bans:
                        print(f"        {champ}: {count} bans")
            
                # Print sample draft sequence
                if results['draft_sequences'][matchup]:
                    print("\n  Sample Draft Sequence:")
                    sample_draft = results['draft_sequences'][matchup][0]  # Show first draft sequence
                    print("    Blue Team:")
                    print(f"      First String Bans: {sample_draft['blue']['fs_bans']}")
                    print(f"      Second String Bans: {sample_draft['blue']['ss_bans']}")
                    print(f"      Picks: {sample_draft['blue']['picks']}")
                    print("    Red Team:")
                    print(f"      First String Bans: {sample_draft['red']['fs_bans']}")
                    print(f"      Second String Bans: {sample_draft['red']['ss_bans']}")
                    print(f"      Picks: {sample_draft['red']['picks']}")
                print()
        
        # Handle series results
        print("\nSeries Results:")
        for length in results['series_results']:
            print(f"\nBest-of-{length}:")
            for matchup in ['MLP_vs_MLP', 'MLP_vs_RNN', 'RNN_vs_MLP', 'RNN_vs_RNN',
                            'MLP_vs_Heuristic', 'Heuristic_vs_MLP', 'RNN_vs_Heuristic', 'Heuristic_vs_RNN', 'Heuristic_vs_Heuristic']:
                print(f"  {matchup}:")
                print(f"    Blue wins: {results['series_results'][length][matchup]['blue']}")
                print(f"    Red wins: {results['series_results'][length][matchup]['red']}")
        
        # Print series_game_breakdown
        if 'series_game_breakdown' in results:
            print("\nSeries Game Winner Sequences:")
            for length, sequences in results['series_game_breakdown'].items():
                print(f"Best-of-{length}:")
                for seq in sequences:
                    print(f"  {seq}")
    else:
        run(best_of=args.best_of)
    args = parser.parse_args()

    with open('matchup_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    if args.matchups:
        print("Matchup Simulation Results:\n")
        # Handle non-series results
        for matchup in ['MLP_vs_MLP', 'MLP_vs_RNN', 'RNN_vs_MLP', 'RNN_vs_RNN',
                        'MLP_vs_Heuristic', 'Heuristic_vs_MLP', 'RNN_vs_Heuristic', 'Heuristic_vs_RNN', 'Heuristic_vs_Heuristic']:
            blue_model, red_model = matchup.split('_vs_')
            print(f"{matchup}:")
            print(f"  {blue_model} wins: {results[matchup]['blue']['wins']} (Game wins: {results[matchup]['blue']['game_wins']})")
            print(f"  {red_model} wins: {results[matchup]['red']['wins']} (Game wins: {results[matchup]['red']['game_wins']})")
            print("  Top 5 picks for each team:")
            for team, team_data in results[matchup].items():
                print(f"    {team} team:")
                
                # Print top picks
                print("      Top picks:")
                top_picks = sorted(team_data['picks'].items(), key=lambda x: x[1], reverse=True)[:5]
                for champ, count in top_picks:
                    print(f"        {champ}: {count} picks")
                
                # Print top bans
                print("      Top bans:")
                top_bans = sorted(team_data['bans'].items(), key=lambda x: x[1], reverse=True)[:5]
                for champ, count in top_bans:
                    print(f"        {champ}: {count} bans")
            
            # Print sample draft sequence
            if results['draft_sequences'][matchup]:
                print("\n  Sample Draft Sequence:")
                sample_draft = results['draft_sequences'][matchup][0]  # Show first draft sequence
                print("    Blue Team:")
                print(f"      First String Bans: {sample_draft['blue']['fs_bans']}")
                print(f"      Second String Bans: {sample_draft['blue']['ss_bans']}")
                print(f"      Picks: {sample_draft['blue']['picks']}")
                print("    Red Team:")
                print(f"      First String Bans: {sample_draft['red']['fs_bans']}")
                print(f"      Second String Bans: {sample_draft['red']['ss_bans']}")
                print(f"      Picks: {sample_draft['red']['picks']}")
            print()
        
        # Handle series results
        print("\nSeries Results:")
        for length in results['series_results']:
            print(f"\nBest-of-{length}:")
            for matchup in ['MLP_vs_MLP', 'MLP_vs_RNN', 'RNN_vs_MLP', 'RNN_vs_RNN',
                            'MLP_vs_Heuristic', 'Heuristic_vs_MLP', 'RNN_vs_Heuristic', 'Heuristic_vs_RNN', 'Heuristic_vs_Heuristic']:
                print(f"  {matchup}:")
                print(f"    Blue wins: {results['series_results'][length][matchup]['blue']}")
                print(f"    Red wins: {results['series_results'][length][matchup]['red']}")
        
        # Print series_game_breakdown
        if 'series_game_breakdown' in results:
            print("\nSeries Game Winner Sequences:")
            for length, sequences in results['series_game_breakdown'].items():
                print(f"Best-of-{length}:")
                for seq in sequences:
                    print(f"  {seq}")
    else:
        run(best_of=args.best_of)
    args = parser.parse_args()
    run(best_of=args.best_of)
