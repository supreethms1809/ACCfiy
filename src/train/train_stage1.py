from src.model.qwen_model import *
from src.dataset.qwen_dataset import *
from trl import SFTTrainer, SFTConfig
import datetime
import json
dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def train_stage1(config, accelerator):
    print("Training stage 1") if accelerator.is_main_process else None
    ds_check = False
    decoder1, model_config, tokenizer = load_decoder1(config)
    # print(f"decoder1: {decoder1}") if accelerator.is_main_process else None
    # print(f"model_config: {model_config}") if accelerator.is_main_process else None
    # print(f"tokenizer: {tokenizer}") if accelerator.is_main_process else None
    dataset = accfiyDataset(config, tokenizer)
    ds = dataset.load_dataset()
    ds = dataset.tokenize_dataset(ds)
    ds_check = dataset.check_batch_size(ds)
    accelerator.wait_for_everyone()

    if ds_check:
        print("Batch size check passed") if accelerator.is_main_process else None
    else:
        print("Batch size check failed") if accelerator.is_main_process else None

    ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_ds = ds["train"]
    eval_ds = ds["test"]
    ds_config_file = f"{config.training_config.stage1.deepspeed_config_file}"
    with open(ds_config_file, 'r') as f:
        ds_config = json.load(f)
    print(f"Deepspeed config file: {ds_config}") if accelerator.is_main_process else None

    sft_config = SFTConfig(
        per_device_train_batch_size=config.training_config.stage1.train_batch_size,
        per_device_eval_batch_size=config.training_config.stage1.eval_batch_size,
        gradient_accumulation_steps=config.training_config.stage1.gradient_accumulation_steps,
        num_train_epochs=config.training_config.stage1.num_epochs,
        torch_compile=config.training_config.stage1.torch_compile,
        deepspeed=ds_config if config.model_config.use_deepspeed.use_zero3 else None,
        fp16=config.training_config.stage1.fp16,
        bf16=config.training_config.stage1.bf16,
        gradient_checkpointing=config.training_config.stage1.gradient_checkpointing,
        logging_steps=config.training_config.stage1.logging_steps,
        save_steps=config.training_config.stage1.save_steps,
        save_total_limit=config.training_config.stage1.save_total_limit,
        run_name=f"{config.training_config.stage1.run_name}-{dt}",
        output_dir=f"{config.training_config.stage1.output_dir}",
        save_safetensors=config.training_config.stage1.save_safetensors,
        save_only_model=config.training_config.stage1.save_only_model,
        max_grad_norm=config.training_config.stage1.max_grad_norm,
        max_seq_length=config.training_config.stage1.max_seq_length,
        save_strategy=config.training_config.stage1.save_strategy,
        eval_strategy=config.training_config.stage1.eval_strategy,
        report_to=config.training_config.stage1.report_to,
        dataset_text_field="input_ids",
    )

    trainer = SFTTrainer(
        model=decoder1,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
        processing_class=tokenizer,
    )
    
    if config.training_stage_config.train_stage1:
        print("Starting training stage 1") if accelerator.is_main_process else None
        if config.training_config.stage1.resume_from_checkpoint:
            print("Resuming training stage 1") if accelerator.is_main_process else None
            trainer.train(resume_from_checkpoint=True)
            print("Training stage 1 completed") if accelerator.is_main_process else None
        else:
            print("Starting training stage 1 from scratch") if accelerator.is_main_process else None
            trainer.train()
            print("Training stage 1 completed") if accelerator.is_main_process else None
        trainer.save_model(config.training_config.stage1.output_dir)
        print(f"Model saved to {config.training_config.stage1.output_dir}") if accelerator.is_main_process else None
    else:
        print("Training stage 1 not enabled") if accelerator.is_main_process else None

    output_dir = config.training_config.stage1.output_dir
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        print(f"Latest checkpoint found: {latest_checkpoint}")
        decoder1_path = os.path.join(output_dir, latest_checkpoint)
    else:
        print("No checkpoints found in output directory") if accelerator.is_main_process else None
        decoder1_path = output_dir

    return decoder1_path