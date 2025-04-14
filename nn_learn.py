import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

import nn_preprocessing

NUM_CHAMPIONS = 170
NUM_ROLES = 5
DRAFT_LENGTH = 20

INPUT_SIZE = NUM_CHAMPIONS * DRAFT_LENGTH
ROLE_NAMES = ["top", "jungle", "mid", "bot", "support"]

draft_df = nn_preprocessing.load_full_draft_csv() # all drafts dataframe
player_data = nn_preprocessing.load_player_data() # player dict[str, pd.DataFrame], sorted by player->champ
pickban_df = nn_preprocessing.load_pickban_data() # pickban dict[str, pd.DataFrame], sorted by patch

class MLPDraftModel(nn.Module):
    """ Multi-Layer Perceptron approach to draft prediction

    Args:
        nn (nn.Module): Inherited from torch.nn.Module
    """
    def __init__(self, input_size, hidden_sizes, output_size=NUM_CHAMPIONS) -> None:
        hidden_sizes = hidden_sizes if hidden_sizes else [512, 256]
        super(MLPDraftModel, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.network(x)

class RNNDraftModel(nn.Module):
    """
    Recurrent Neural Network approach to draft prediction

    Args:
        nn (nn.Module): Inherited from torch.nn.Module
    """
    def __init__(self, input_size=NUM_CHAMPIONS, hidden_size=256, 
        num_layers=2, output_size=NUM_CHAMPIONS):
        super(RNNDraftModel, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        out, _ =self.rnn(x)
        return self.fc(out[:, -1, :])

def eval_model(model, data_loader, device="cpu"):
    """
    Evaluate the model on the given data loader

    Args:
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): The data loader
        device (str, optional): The device to run the model on. Defaults to "cpu".

    Returns:
        tuple: A tuple of accuracy and F1 score
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

def encode_draft_state(picks_bans, player_roles, patch_vec=None):
    """
    Encode the draft state into a feature vector

    Args:
        picks_bans (list): List of picks and bans
        player_roles (dict): Dictionary of player roles
        patch_vec (torch.Tensor, optional): Patch features. Defaults to None.

    Returns:
        torch.Tensor: Encoded draft state
    """
    one_hot = torch.zeros(NUM_CHAMPIONS * DRAFT_LENGTH)
    for i, champ in enumerate(picks_bans):
        if champ != -1:
            one_hot[i * NUM_CHAMPIONS + champ] = 1

    player_features = torch.cat([player_roles[role] for role in ROLE_NAMES])
    if patch_vec is not None:
        return torch.cat([one_hot, player_features, patch_vec])
    else:
        return torch.cat([one_hot, player_features])

def build_training_data_from_drafts(p_draft_df, p_player_data, p_pickban_df):
    """
    Convert drafts into supervised learning format: (state, next_action)

    Args:
        draft_df (pd.DataFrame): DataFrame containing draft data
        player_data (dict): Dictionary of player data
        pickban_df (pd.DataFrame): DataFrame containing pickban data

    Returns:
        tuple: A tuple of (X, y) where X is the feature matrix and y is the target vector
    """
    X, y = [], []
    for _, row in p_draft_df.iterrows():
        picks_bans = [-1] * DRAFT_LENGTH
        player_roles = {role: torch.tensor(p_player_data[row[role]]['features'].values, 
            dtype=torch.float32) for role in ROLE_NAMES}
        patch_vec = torch.tensor(p_pickban_df.loc[row['patch']]['features'].values, 
            dtype=torch.float32)

        for i in range(DRAFT_LENGTH):
            champ = row[f'step_{i}']
            x_vec = encode_draft_state(picks_bans, player_roles, patch_vec)
            X.append(x_vec)
            y.append(champ)
            picks_bans[i] = champ
    return torch.stack(X), torch.tensor(y, dtype=torch.long)
