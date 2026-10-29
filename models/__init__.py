from .base import BaseModel

MODEL_REGISTRY = {"base": BaseModel}


def build_model(model_config):
    model_name = model_config["name"]
    model_class = MODEL_REGISTRY[model_name]

    return model_class()
