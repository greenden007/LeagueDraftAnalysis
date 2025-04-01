"""
Neural Network Preprocessing, Training, and Construction for utilization
"""
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split


def get_csv_tournament(file: str):
    df = pd.read_csv(f"tournament_draft_csvs/{file}")
    return df

def get_csv_pickban(file: str):
    df = pd.read_csv(f"PickBan_csvs/{file}")
    return df

# Do not use if possible!
# Look to load from file instead
def preprocess_draft_data(df: pd.DataFrame):
    processed_bans = []
    g_df = df.groupby(df.index // 6)
    for _, group in g_df:
        blue_fs_bans = group["blue_bans"].tolist()[:3]
        blue_picks = group["blue_picks"].tolist()[:5]
        red_fs_bans = group["red_bans"].tolist()[:3]
        red_picks = group["red_picks"].tolist()[:5]
        blue_ss_bans = group["blue_bans"].tolist()[-2:]
        red_ss_bans = group["red_bans"].tolist()[-2:]
        blue_players = group["blue_roster"].iloc[:5]
        red_players = group["red_roster"].iloc[:5]
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

# Avoid using this function if possible
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
    g_df = pickban_df.groupby("patch")
    return {patch: g_df.get_group(patch) for patch in g_df.groups}

def build_draft_df():
    full_df = pd.DataFrame()
    for file in os.listdir("tournament_draft_csvs"):
        df = get_csv_tournament(file)
        df = preprocess_draft_data(df)
        clean_ban_data(df)
        full_df = pd.concat([full_df, df], ignore_index=True)
    return full_df

class LOLDraftModel:
    """
    A neural network model for draft prediction.
    """
    def __init__(self, num_champions, embedding_dim=64, lstm_units=128):
        self.num_champions = num_champions
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = self._build_model()

    def _build_model(self):
        current_draft = layers.Input(shape=(10,), name="current_draft")
        side = layers.Input(shape=(1,), name="side")
        meta_relevance = layers.Input(shape=(self.num_champions,), name="meta_relevance")
        player_comfort = layers.Input(shape=(self.num_champions,), name="player_comfort")
        player_id = layers.Input(shape=(1,), name="player_id")
        team_id = layers.Input(shape=(1,), name="team_id")
        phase = layers.Input(shape=(1,), name="draft_phase")

        champ_embedding = layers.Embedding(self.num_champions + 1, self.embedding_dim, mask_zero=True)
        team_embedding = layers.Embedding(100, 32, name="team_embedding")(team_id)
        player_embedding = layers.Embedding(1000, 32, name="player_embedding")(player_id)

        draft_embedded = champ_embedding(current_draft)
        draft_lstm = layers.LSTM(self.lstm_units)(draft_embedded)

        side_features = layers.Dense(16, activation="relu")(side)
        meta_dense = layers.Dense(16, activation="relu")(meta_relevance)
        player_comfort_dense = layers.Dense(16, activation="relu")(player_comfort)

        combined = layers.concatenate({
            draft_lstm,
            side_features,
            meta_dense,
            player_comfort_dense,
            team_embedding,
            player_embedding,
            phase
        })

        # Deep Layers
        x = layers.Dense(256, activation="relu")(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(self.num_champions, activation="softmax")(x)

        model = Model(
            inputs=[
                current_draft,
                side,
                meta_relevance,
                player_comfort,
                player_id,
                team_id,
                phase
            ],
            outputs=outputs
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def train(self, train_data, validation_data, epochs=20, batch_size=64):
        return self.model.fit(train_data, validation_data, epochs=epochs, batch_size=batch_size)
    
    def predict_next_pick(self, current_draft, side, meta_relevance, player_comfort, player_id, team_id, draft_phase):
        prediction = self.model.predict([
            np.array([current_draft]),
            np.array([side]),
            np.array([meta_relevance]),
            np.array([player_comfort]),
            np.array([player_id]),
            np.array([team_id]),
            np.array([draft_phase])
        ])

        top_k = 5
        top_indices = np.argsort(prediction[0])[-top_k:][::-1]
        top_probs = prediction[0][top_indices]

        return list(zip(top_indices, top_probs))
    
    def save(self, file: str):
        self.model.save(file)

    @classmethod
    def load(cls, filepath, num_champions):
        instance = cls(num_champions)
        instance.model = tf.keras.models.load_model(filepath)
        return instance

def load_full_draft_csv() -> pd.DataFrame:
    return pd.read_csv("processed_data_files/full_draft_df.csv")

def load_player_data() -> dict[str, pd.DataFrame]:
    return pickle.load(open("processed_data_files/player_data.pkl", "rb"))

def load_pickban_data() -> dict[str, pd.DataFrame]:
    return pickle.load(open("processed_data_files/pickban_df.pkl", "rb"))

def get_unique_players(df: pd.DataFrame) -> list[str]:
    """
    Returns a list of unique player names from the entire draft data.

    Args:
        df (pd.DataFrame): The DataFrame containing every draft's data.

    Returns:
        list[str]: A list of unique player names.
    """
    players = set()
    for index, row in df.iterrows():
        for player in row["blue_roster"] + row["red_roster"]:
            players.add(player)
    return list(players)


def organize_by_player_performance(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Organizes the draft data by player performance.

    Args:
        df (pd.DataFrame): The DataFrame containing the draft data.

    Returns:
        dict[str, pd.DataFrame]: A dictionary mapping player names to their draft data and win rate on champion.
    """

    player_data = {}
    for index, row in df.iterrows():
        for player in row["blue_players"]:
            champion = row["blue_picks"][row["blue_players"].to_list().index(player)]
            if player not in player_data:
                player_data[player] = {}
            if champion not in player_data[player]:
                player_data[player][champion] = {"wins": 0, "games": 0}
            if row["winner"] == 1.0:
                player_data[player][champion]["wins"] += 1
            player_data[player][champion]["games"] += 1

        for player in row["red_players"]:
            champion = row["red_picks"][row["red_players"].to_list().index(player)]
            if player not in player_data:
                player_data[player] = {}
            if champion not in player_data[player]:
                player_data[player][champion] = {"wins": 0, "games": 0}
            if row["winner"] == 0.0:
                player_data[player][champion]["wins"] += 1
            player_data[player][champion]["games"] += 1
    
    for player, champions in player_data.items():
        for champion, stats in champions.items():
            stats["win_rate"] = stats["wins"] / stats["games"]
            player_data[player][champion] = pd.DataFrame([stats])
    
    return player_data
        

def main():
    """
    Main function to build pickban and draft dataframes.
    """

    draft_df = load_full_draft_csv()
    # player_data = load_player_data()
    # pickban_df = load_pickban_data()

    print(draft_df)
    # print(player_data)
    # print(pickban_df)
    print(draft_df.columns)

    df_draft_confirm = build_draft_df()


if __name__ == "__main__":
    main()
