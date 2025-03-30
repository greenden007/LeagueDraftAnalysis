import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

def get_csv_tournament(file: str):
    df = pd.read_csv(f"tournament_draft_csvs/{file}")
    return df

def get_csv_pickban(file: str):
    df = pd.read_csv(f"PickBan_csvs/{file}")
    return df

def preprocess_draft_data(df: pd.DataFrame):
    processed_bans = []
    g_df = df.groupby(df.index // 6)
    print(g_df)
    for _, group in g_df:
        blue_fs_bans = group["blue_bans"].tolist()[:3]
        blue_picks = group["blue_picks"].tolist()[:5]
        red_fs_bans = group["red_bans"].tolist()[:3]
        red_picks = group["red_picks"].tolist()[:5]
        blue_ss_bans = group["blue_bans"].tolist()[-2:]
        red_ss_bans = group["red_bans"].tolist()[-2:]
        blue_players = group["Blue Side Roster"].iloc[:5]
        red_players = group["Red Side Roster"].iloc[:5]
        patch = group["patch"].iloc[0]
        blue_side = group["blue_side"].iloc[5]
        red_side = group["red_side"].iloc[5]
        winner = group["winner"].iloc[5]
        processed_bans.append({
            "blue_fs_bans": blue_fs_bans,
            "red_fs_bans": red_fs_bans,
            "blue_picks": blue_picks,
            "red_picks": red_picks,
            "blue_ss_bans": blue_ss_bans,
            "red_ss_bans": red_ss_bans,
            "blue_players": blue_players,
            "red_players": red_players,
            "patch": patch,
            "blue_side": blue_side,
            "red_side": red_side,
            "winner": winner
        })
    return pd.DataFrame(processed_bans)

def clean_ban_data(df: pd.DataFrame):
    """
    Cleans the ban data by converting the winner column to 1.0 and 0.0.

    Args:
        df (pd.DataFrame): The DataFrame containing the ban data.
    """
    df["winner"] = df["winner"].map({"blue_side": 1.0, "red_side": 0.0})
    if df["winner"].isna().any():
        raise ValueError("Invalid value found in winner column. Expected 'blue_side' or 'red_side'")

def build_pickban_df():
    pickban_df = pd.DataFrame()
    for file in os.listdir("PickBan_csvs"):
        df = get_csv_pickban(file)
        pickban_df = pd.concat([pickban_df, df], ignore_index=True)
    return pickban_df

def build_draft_df():
    full_df = pd.DataFrame()
    for file in os.listdir("tournament_draft_csvs"):
        df = get_csv_tournament(file)
        df = preprocess_draft_data(df)
        clean_ban_data(df)
        full_df = pd.concat([full_df, df], ignore_index=True)
    return full_df

class DraftMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DraftMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_model(matches: pd.DataFrame, pb_percent: pd.DataFrame):
    pass
    # encoder = OneHotEncoder(sparse=False)
    # X = matches.values # feature matrix
    # y = pb_percent.values # label matrix

    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # y_train = torch.tensor(y_train, dtype=torch.float32)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)

    # input_size = X_train.shape[1]
    # hidden_size = 128
    # output_size = 1

    # model = DraftMLP(input_size, hidden_size, output_size)
    
    # criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # epochs = 10
    # for epoch in range(epochs):
    #     model.train()
    #     optimizer.zero_grad()
        
    #     outputs = model(X_train)
    #     loss = criterion(outputs, y_train)
    #     loss.backward()
    #     optimizer.step()

    #     if (epoch + 1) % 10 == 0:
    #         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")
    
    # model.eval()
    # with torch.no_grad():
    #     y_pred = model(X_test)
        

def main():
    draft_df = build_draft_df()
    pickban_df = build_pickban_df()
    print(draft_df)
    print(pickban_df)

    train_model(draft_df, pickban_df)
if __name__ == "__main__":
    main()
