from src.model.qwen_model import *
from src.train.train_stage1 import *
from src.train.train_stage2 import *
import os


def train_model(config, accelerator):
    if config.training_stage_config.train_stage1:
        decoder1_path = train_stage1(config, accelerator)
        try:
            decoder1_path = config.combined_model.combined_model_decoder1_local_path
            if not os.path.exists(decoder1_path):
                raise ValueError(f"Decoder1 path {decoder1_path} does not exist")
            else:
                print(f"Decoder1 path {decoder1_path} exists")
        except:
            ValueError("Decoder1 path not found")
    else:
        decoder1_path = None

    if config.training_stage_config.train_stage2:
        try:
            decoder1_path = get_decoder1_path(config)
            print("Decoder1 path found: ", decoder1_path) if accelerator.is_main_process else None
        except:
            ValueError("Decoder1 path not found. Initializing from scratch")
        print("Training stage 2, decoder1_path: ", decoder1_path) if accelerator.is_main_process else None
        decoder2_path = None
        mapper_path = None
        combined_model, tokenizer, model_config = load_combined_model(config, decoder1_path, decoder2_path, mapper_path)
        train_stage2(config, accelerator, tokenizer, combined_model, model_config)


