import argparse
from src.utils.common import read_config
from src.train.train_driver import train_model
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from accelerate import Accelerator, DeepSpeedPlugin

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./src/model_config/qwen_config.yaml")
    parser.add_argument("--model_name", type=str, default="qwen")
    args = parser.parse_args()

    config = read_config(args.config)
    if config.model_config.use_deepspeed.use_zero3: 
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=config.training_config.stage1.deepspeed_config_file)
        accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
    else:
        accelerator = Accelerator()
    print(f"Config: {config}") if accelerator.is_main_process else None

    # Check training or evaluation
    if config.training_stage_config.train_model:
        print("Training model for stages: ") if accelerator.is_main_process else None
        if config.training_stage_config.train_stage1:
            print("Training stage 1") if accelerator.is_main_process else None
        if config.training_stage_config.train_stage2:
            print("Training stage 2") if accelerator.is_main_process else None
        if config.training_stage_config.train_stage3:
            print("Training stage 3") if accelerator.is_main_process else None
        if config.training_stage_config.train_stage4:
            print("Training stage 4") if accelerator.is_main_process else None

        train_model(config, accelerator)

    if config.evaluation_stage_config.eval_model:
        print("Evaluating model for stages: ") if accelerator.is_main_process else None
        if config.evaluation_stage_config.eval_stage1:
            print("Evaluating stage 1") if accelerator.is_main_process else None
        if config.evaluation_stage_config.eval_stage2:
            print("Evaluating stage 2") if accelerator.is_main_process else None
        if config.evaluation_stage_config.eval_stage3:
            print("Evaluating stage 3") if accelerator.is_main_process else None
        if config.evaluation_stage_config.eval_stage4:
            print("Evaluating stage 4") if accelerator.is_main_process else None

    time.sleep(1000)
    print(f"Successfully Completed everything") if accelerator.is_main_process else None
    print("-"*50) if accelerator.is_main_process else None


if __name__ == "__main__":
    main()