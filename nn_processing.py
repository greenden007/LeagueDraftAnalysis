import nn_preprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

draft_df = nn_preprocessing.load_full_draft_csv()
player_data = nn_preprocessing.load_player_data()
pickban_df = nn_preprocessing.load_pickban_data()

class LoLDraftRNN:
    def __init__(self, num_champions=170, embedding_dim=64, lstm_units=128, player_features_dim=0, patch_features_dim=0):
        self.num_champions = num_champions
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.player_features_dim = player_features_dim
        self.patch_features_dim = patch_features_dim
        
        self.vocab_size = num_champions + 1 # +1 for no ban situations
        self.BLUE_TEAM = 0
        self.RED_TEAM = 1
        self.BAN_ACTION = 0
        self.PICK_ACTION = 1

        self.series_picked_champions = set()
        self.model = self._build_model()
    
    def _build_model(self):
        draft_seq_input = Input(shape=(None,), name='draft_seq')
        action_type_input = Input(shape=(None, 1), name='action_type')
        team_input = Input(shape=(None, 1), name='team')

        availability_mask_input = Input(shape=(self.num_champions,), name='availability_mask')
        player_inputs = []
        if self.player_features_dim > 0:
            for role in ["top", "jungle", "mid", "bot", "support"]:
                for team in ["blue", "red"]:
                    player_inputs.append(Input(shape=(self.patch_features_dim,), name=f'{team}_{role}_features'))
        
        patch_input = None
        if self.patch_features_dim > 0:
            patch_input = Input(shape=(self.patch_features_dim,), name='patch_features')
        
        champion_embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='champion_embedding'
        )(draft_seq_input)

        draft_features = Concatenate(axis=-1)([
            champion_embedding,
            action_type_input,
            team_input
        ])

        masked_features = Masking(mask_value=0.0)(draft_features)
        lstm_output = LSTM(self.lstm_units, return_sequences=False)(masked_features)

        if player_inputs:
            all_player_features = Concatenate(axis=-1)(player_inputs)
            player_features_processed = Dense(128, activation='relu')(all_player_features)
            lstm_output = Concatenate()([lstm_output, player_features_processed])
        
        if patch_input is not None:
            patch_features_processed = Dense(64, activation='relu')(patch_input)
            lstm_output = Concatenate()([lstm_output, patch_features_processed])
        
        # Final layers
        x = Dense(256, activation='relu')(lstm_output)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        champion_logits = Dense(self.num_champions)(x)
        masked_logits = champion_logits * availability_mask_input
        
        champion_probabilities = tf.keras.layers.Softmax()(masked_logits)
        inputs = [draft_seq_input, action_type_input, team_input, availability_mask_input]
        if player_inputs:
            inputs.extend(player_inputs)
        if patch_input is not None:
            inputs.append(patch_input)
        
        model = Model(inputs, outputs=champion_probabilities)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def prep_data(self, draft_df, player_data, pickban_df):
        pass # TODO: Finish later

    def train(self, X, y, validation_data=None, epochs=10, batch_size=32):
        return self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
    
    def predict_next(self, current_draft, team, action_type, series_picked=None, player_features=None, patch_features=None):
        pass # TODO: Finish later
    
    def create_availability_mask(self, current_draft, series_picked=None):
        mask = np.ones(self.num_champions, dtype=np.float32)
        for champ in current_draft:
            if champ > 0 and champ <= self.num_champions:
                mask[champ-1] = 0
        
        if series_picked:
            for champ in series_picked:
                if champ > 0 and champ <= self.num_champions:
                    mask[champ-1] = 0
        return mask
    
def process_draft_data(matches_df, champion_data):
    pass # TODO: Finish later

def extract_player_features(player_data, champion_data):
    pass # TODO: Finish later

def extract_patch_features(patch_data, champion_data):
    pass # TODO: Finish later

def process_data_patch(patch_data, champion_data):
        pass # TODO: Finish later

def main():
    pass

if __name__ == "__main__":
    main()
