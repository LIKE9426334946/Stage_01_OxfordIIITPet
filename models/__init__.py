from .base import BaseModel
from .heart01 import HeartNetwork_01

MODEL_REGISTRY = {"base": BaseModel, "heart01": HeartNetwork_01}


def build_model(model_config):
    model_name = model_config["name"]
    model_class = MODEL_REGISTRY[model_name]
    kwargs = {"in_channels": 3, "out_channels": 1}

    return model_class(**kwargs)
