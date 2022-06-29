# Load libraries ---------------------------------------------
import torch
import numpy as np
import pandas as pd
# ------------------------------------------------------------

class RecommenderToolkit(object):

  @staticmethod
  def get_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  @staticmethod
  def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)

  @staticmethod
  def generate_negative_interactions(interactions_df, rng, n_users = 0, n_items = 0, n_neg_per_pos = 5):
    negative_interactions = []

    r = np.zeros(shape=(n_users, n_items))
    for _, interaction in interactions_df.iterrows():
      r[int(interaction['user_id'])][int(interaction['item_id'])] = 1

    i = 0
    while i < n_neg_per_pos * len(interactions_df):
      sample_size = 1000
      user_ids = rng.choice(np.arange(n_users), size=sample_size)
      item_ids = rng.choice(np.arange(n_items), size=sample_size)

      j = 0
      while j < sample_size and i < n_neg_per_pos * len(interactions_df):
        if r[user_ids[j]][item_ids[j]] == 0:
          negative_interactions.append([user_ids[j], item_ids[j], 0])
          i += 1
        j += 1

    return negative_interactions

  @staticmethod
  def generate_alternate_negative_interactions(interactions_df: pd.DataFrame, items_df: pd.DataFrame, rng, n_neg_per_pos = 5):
    # Generate n_neg_per_pos alternate items
    alternate_items = dict()

    values = items_df.to_numpy()

    for idx, row in zip(range(len(values)), values):
      j = 0
      infinite_loop_cap = 0

      while j < n_neg_per_pos and infinite_loop_cap < 1000:
        infinite_loop_cap += 1
        random_index = rng.randint(0, len(values) - 1)
        compare_row = values[random_index]

        if 2 not in compare_row + row:
          if idx not in alternate_items:
            alternate_items[idx] = []

          alternate_items[idx].append(random_index)
          j += 1
    
    negative_interactions = []

    for idx, interaction in interactions_df.iterrows():
      current_item_id = interaction['item_id']
      if current_item_id >= len(values):
        current_item_id = rng.randint(0, len(values) - 1)

      alternate_item_ids = values[interaction['item_id']]

      for item_id in alternate_item_ids:
        negative_interactions.append([interaction['user_id'], item_id, 0])

    return negative_interactions