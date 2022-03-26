"""
This module offers a method to create a config object for each model.
This will be saved by pytorch lighning inside the parameters.yml, so you
can see all the parameters in tensorboard.
"""

dataset_default_config = {
    # data module options
    "column_goldlabel": "score",
    "dataset_path": "../../data/USParties/USParties_preprocessed.csv",
    "category_group_id": True,  # encodes the group id into the category one-hot-vector
    "category_type": True,  # encodes the post type into the category one-hot-vector
    "category_tld": True,  #  encodes the links top level domain (tld) into the category one-hot-vector
    "batch_size": 128,
    # model options
    "category_encoded_length": 16,  # length of the one hot encoding vector
    "category_encoder_out": 16,  # output dim of the category encoder layer
    "learning_rate": 1e-3,
    # BiLSTM dataset preprocessing
    "vocab_min_freq": 10,  # every word below this frequncy will not be added to the vocab
    "bilstm_hidden_dim": 150,
    "embedding_dim": 100,
}


def create_config(config):
    return {**dataset_default_config, **config}
