import pandas as pd
import torch
from nn_learn import PAD_IDX, DRAFT_LENGTH, MLPDraftModel, RNNDraftModel, encode_sequence, champ2idx, meta_vectors, compute_comfort

# Initialize models
vocab_size = len(champ2idx) + 1
mlp = MLPDraftModel(vocab_size=vocab_size)
rnn = RNNDraftModel(vocab_size=vocab_size)

mlp.load_state_dict(torch.load("mlp_model.pt"))
rnn.load_state_dict(torch.load("rnn_model.pt"))
mlp.eval()
rnn.eval()

# Example partial draft (replace with real data)
partial_row = pd.Series({
    'patch': '14.1',
    'blue_fs_bans': ['Aatrox','Blitzcrank','Caitlyn'],
    'red_fs_bans':  ['Darius','Ekko','Fiora'],
    'blue_picks':   ['Garen','Hecarim','Irelia','Jinx','Karma'],
    'red_picks':    ['Lulu','Maokai','Nidalee','Orianna','Pyke'],
    'blue_ss_bans': ['Leona','Morgana'],
    'red_ss_bans':  ['Nautilus','Olaf'],
    'blue_players':['P1','P2','P3','P4','P5']
})
seq = encode_sequence(partial_row)
seq = seq + [PAD_IDX] * (DRAFT_LENGTH - len(seq))
meta = meta_vectors[partial_row['patch']]
comfort = compute_comfort(partial_row['blue_players'])

next_mlp = mlp.predict_next(seq, meta, comfort)
next_rnn = rnn.predict_next(seq, meta, comfort)

print("MLP suggests:", next_mlp)
print("RNN suggests:", next_rnn)

