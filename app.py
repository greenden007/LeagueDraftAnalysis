from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from nn_learn import MLPDraftModel, RNNDraftModel, champ2idx, idx2champ, compute_comfort, meta_vectors
import json

app = Flask(__name__)
CORS(app)

vocab_size = len(champ2idx) + 1
mlp = MLPDraftModel(vocab_size=vocab_size)
rnn = RNNDraftModel(vocab_size=vocab_size)
mlp.load_state_dict(torch.load("mlp_model.pt"))
rnn.load_state_dict(torch.load("rnn_model.pt"))
mlp.eval()
rnn.eval()

with open('champion_roles.json') as f:
    champion_roles = json.load(f)
champions = list(champ2idx.keys())

@app.route('/api/champions', methods=['GET'])
def get_champions():
    return jsonify(champions)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    draft_sequence = data.get('draft_sequence', [])
    player_roles = data.get('player_roles', {})
    patch = data.get('patch', '14.12')
    model = data.get('model', 'rnn')
    
    seq = [champ2idx.get(champ, 0) for champ in draft_sequence]
    meta = meta_vectors.get(patch, torch.zeros(vocab_size))
    comfort = compute_comfort(list(player_roles.keys()))

    pred = None
    mlp_probs = None
    rnn_probs = None
    with torch.no_grad():
        if model == 'mlp':
            mlp_input = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
            mlp_logits = mlp(mlp_input, meta.unsqueeze(0), comfort.unsqueeze(0), torch.zeros_like(meta).unsqueeze(0))
            mlp_probs = torch.softmax(mlp_logits, dim=1)
            mlp_pred = torch.argmax(mlp_logits, dim=1).item()
            pred = idx2champ[mlp_pred]
        elif model == 'rnn':
            rnn_input = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
            rnn_logits = rnn(rnn_input, meta.unsqueeze(0), comfort.unsqueeze(0), torch.zeros_like(meta).unsqueeze(0))
            rnn_probs = torch.softmax(rnn_logits, dim=1)
            rnn_pred = torch.argmax(rnn_logits, dim=1).item()
            pred = idx2champ[rnn_pred]
    
    response = {
        'champion': pred,
        'probability': float(mlp_probs[mlp_pred]) if model == 'mlp' else float(rnn_probs[rnn_pred])
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
