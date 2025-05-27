import os
from omegaconf import OmegaConf

def read_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf