import nn_preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import itertools

NUM_CHAMPIONS = 170
NUM_ROLES = 5
DRAFT_LENGTH = 20

INPUT_SIZE = NUM_CHAMPIONS * DRAFT_LENGTH
ROLE_NAMES = ["top", "jungle", "mid", "bot", "support"]

draft_df = nn_preprocessing.load_full_draft_data()  # all drafts dataframe
player_data = nn_preprocessing.load_player_data()  # player dict[str, pd.DataFrame]
pickban_df = nn_preprocessing.load_pickban_data()  # pickban dict[str, pd.DataFrame]
champ_matchup_info = nn_preprocessing.clean_matchup_info()  # dict[str, pd.DataFrame]


