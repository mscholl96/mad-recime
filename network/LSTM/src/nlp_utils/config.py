"""
This module offers a method to create a config object for each model.
This will be saved by pytorch lighning inside the parameters.yml, so you
can see all the parameters in tensorboard.
"""

dataset_default_config = dict(
    epochs=10,
    batchSize=10,
    lr=1e-3,
    ratio=[0.7, 0.3],
    hiddenDim=256,
    numLayers=1,
    embeddingDim=300
)


def create_config(config):
    return {**dataset_default_config, **config}
