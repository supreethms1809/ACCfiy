from src.model.qwen_model import *
from src.dataset.qwen_dataset import *
from trl import SFTTrainer, SFTConfig
import datetime
import json
dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def train_stage2(config, accelerator, tokenizer, combined_model, model_config):
    print("Training stage 2") if accelerator.is_main_process else None
    ds_check = False
    dataset = accfiySyntheticDataset(config, tokenizer)
    ds = dataset.load_dataset()
    ds = dataset.tokenize_dataset(ds)
    ds_check = dataset.check_batch_size(ds)
    accelerator.wait_for_everyone()
    if ds_check:
        print("Dataset check passed") if accelerator.is_main_process else None
    else:
        print("Dataset check failed") if accelerator.is_main_process else None
        return
    
    ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_ds = ds["train"]
    eval_ds = ds["test"]
    
    if config.model_config.use_deepspeed:
        ds_config = config.training_config.stage2.deepspeed_config_file
    else:
        ds_config = None
    
    # sft_config = SFTConfig(
    #     per_device_train_batch_size=config.training_config.stage2.train_batch_size,
    #     per_device_eval_batch_size=config.training_config.stage2.eval_batch_size,
    #     gradient_accumulation_steps=config.training_config.stage2.gradient_accumulation_steps,
    #     num_train_epochs=config.training_config.stage2.num_epochs,
    #     torch_compile=config.training_config.stage2.torch_compile,
    #     deepspeed=ds_config if config.model_config.use_deepspeed.use_zero3 else None,
    #     fp16=config.training_config.stage2.fp16,
    #     bf16=config.training_config.stage2.bf16,
    #     gradient_checkpointing=config.training_config.stage2.gradient_checkpointing,
    #     logging_steps=config.training_config.stage2.logging_steps,
    #     save_steps=config.training_config.stage2.save_steps,
    #     save_total_limit=config.training_config.stage2.save_total_limit,
    #     run_name=f"{config.training_config.stage2.run_name}-{dt}",
    #     output_dir=f"{config.training_config.stage2.output_dir}",
    #     save_safetensors=config.training_config.stage2.save_safetensors,
    #     save_only_model=config.training_config.stage2.save_only_model,
    #     max_grad_norm=config.training_config.stage2.max_grad_norm,
    #     max_seq_length=config.training_config.stage2.max_seq_length,
    #     save_strategy=config.training_config.stage2.save_strategy,
    #     eval_strategy=config.training_config.stage2.eval_strategy,
    #     report_to=config.training_config.stage2.report_to,
    #     dataset_text_field="input_ids",
    # )

    # trainer = SFTTrainer(
    #     model=combined_model,
    #     train_dataset=train_ds,
    #     eval_dataset=eval_ds,
    #     args=sft_config,
    #     processing_class=tokenizer,
    # )
    
    # if config.training_stage_config.train_stage2:
    #     print("Starting training stage 2") if accelerator.is_main_process else None
    #     if config.training_config.stage2.resume_from_checkpoint:
    #         print("Resuming training stage 2") if accelerator.is_main_process else None
    #         trainer.train(resume_from_checkpoint=True)
    #         print("Training stage 2 completed") if accelerator.is_main_process else None
    #     else:
    #         print("Starting training stage 2 from scratch") if accelerator.is_main_process else None
    #         trainer.train()
    #         print("Training stage 2 completed") if accelerator.is_main_process else None
    #     combined_model.save_model(config.training_config.stage2.output_dir)
    #     print(f"Model saved to {config.training_config.stage2.output_dir}") if accelerator.is_main_process else None
    # else:
    #     print("Training stage 2 not enabled") if accelerator.is_main_process else None

    # output_dir = config.training_config.stage2.output_dir
    # checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    # if checkpoints:
    #     latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    #     print(f"Latest checkpoint found: {latest_checkpoint}")
    #     combined_model_path = os.path.join(output_dir, latest_checkpoint)
    # else:
    #     print("No checkpoints found in output directory") if accelerator.is_main_process else None
    #     combined_model_path = output_dir

    # return combined_model_path

    
