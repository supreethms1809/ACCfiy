from src.model.qwen_model import *
from src.dataset.qwen_dataset import *
from trl import SFTTrainer, SFTConfig
from accelerate import PartialState
from datasets import Dataset
from trl.trainer.utils import ConstantLengthDataset
from trl.trainer.sft_trainer import pack_dataset, truncate_dataset
import torch
import datetime
import json
dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class CustomSFTTrainer(SFTTrainer):
    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        packing,
        formatting_func,
        dataset_name,
        **kwargs,
    ):
        # Convert the dataset to an IterableDataset if it is a ConstantLengthDataset
        if isinstance(dataset, ConstantLengthDataset):
            return dataset

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            if not is_processed:
                # Add custom prompt before chat formatting
                def add_custom_prompt(example):
                    if "cpp_code" in example and "cuda_code" in example:
                        example["prompt"] = f"You are a helpful assistant that transforms the CPU code to GPU code. \
                            Your task is to Generate the CUDA kernel function for the corresponding C++ code. \
                            <task> TRANSFORM_C++_TO_CUDA_KERNEL </task>\n<code_cpp>\n{example['cpp_code']}\n</code_cpp>"
                        example["completion"] = f"<kernel>\n{example['cuda_code']}\n</kernel>"
                        
                        # Add decoder1 prompt
                        example["prompt_decoder1"] = f"You are a helpful assistant that analyzes the C++ code and provides an analysis and instructions for parallelization of the code using CUDA. \
                            Your task is to analyze C++ code and provide an analysis and instructions for parallelization of the code using CUDA. \
                            <task> ANALYZE_FOR_PARALLELIZATION </task>\n<code_cpp>\n{example['cpp_code']}\n</code_cpp>"
                    return example

                dataset = dataset.map(add_custom_prompt, **map_kwargs)

                # Apply chat template
                def apply_chat_template_with_decoder1(example):
                    # Format for main model
                    messages = [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": example["completion"]}
                    ]
                    full_prompt = processing_class.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False, 
                        enable_thinking=False
                    )
                    
                    # Format for decoder1
                    messages_decoder1 = [
                        {"role": "user", "content": example["prompt_decoder1"]}
                    ]
                    full_prompt_decoder1 = processing_class.apply_chat_template(
                        messages_decoder1, 
                        tokenize=False, 
                        add_generation_prompt=False, 
                        enable_thinking=False
                    )
                    
                    return {
                        "text": full_prompt,
                        "text_decoder1": full_prompt_decoder1
                    }

                dataset = dataset.map(apply_chat_template_with_decoder1, **map_kwargs)

                # Tokenize both prompts
                def tokenize_both(example):
                    # Tokenize main prompt
                    tokenized = processing_class(
                        text=example["text"],
                        truncation=True,
                        max_length=1024,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    
                    # Tokenize decoder1 prompt
                    tokenized_decoder1 = processing_class(
                        text=example["text_decoder1"],
                        truncation=True,
                        max_length=1024,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    # Ensure all outputs are tensors
                    if not isinstance(tokenized["input_ids"], torch.Tensor):
                        tokenized["input_ids"] = torch.tensor(tokenized["input_ids"])
                    if not isinstance(tokenized["attention_mask"], torch.Tensor):
                        tokenized["attention_mask"] = torch.tensor(tokenized["attention_mask"])
                    if not isinstance(tokenized_decoder1["input_ids"], torch.Tensor):
                        tokenized_decoder1["input_ids"] = torch.tensor(tokenized_decoder1["input_ids"])

                    return {
                        "input_ids": tokenized["input_ids"].squeeze(0),
                        "attention_mask": tokenized["attention_mask"].squeeze(0),
                        "input_ids_decoder1": tokenized_decoder1["input_ids"].squeeze(0),
                        "attention_mask_decoder1": tokenized_decoder1["attention_mask"].squeeze(0),
                        "labels": tokenized["input_ids"].squeeze(0).clone()
                    }

                dataset = dataset.map(tokenize_both, **map_kwargs)

            # Pack or truncate if needed
            if packing and args.max_length is not None:
                dataset = dataset.select_columns(["input_ids", "attention_mask", "input_ids_decoder1", "attention_mask_decoder1", "labels"])
                dataset = pack_dataset(dataset, args.max_length, map_kwargs)
            elif args.max_length is not None:
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
        print(f"dataset: {dataset}")
        
        return dataset


def train_stage2(config, accelerator, tokenizer, combined_model, model_config):
    print("Training stage 2") if accelerator.is_main_process else None
    ds_check = False
    dataset = accfiySyntheticDataset(config, tokenizer)
    ds = dataset.load_dataset()
    # ds = dataset.tokenize_dataset(ds)
    # ds_check = dataset.check_batch_size(ds)
    # accelerator.wait_for_everyone()
    # if ds_check:
    #     print("Dataset check passed") if accelerator.is_main_process else None
    # else:
    #     print("Dataset check failed") if accelerator.is_main_process else None
    #     return
    
    def custom_collate_fn(features):
        """
        Collate function that selects only required fields and converts them to tensors.
        Args:
            features: List of dictionaries containing the features
        Returns:
            Dictionary with batched tensors for input_ids, attention_mask, input_ids_decoder1, and labels
        """
        if not features:
            return {}
            
        # Select only the required fields
        required_fields = ['input_ids', 'attention_mask', 'input_ids_decoder1', 'attention_mask_decoder1', 'labels']
        batch = {}
        
        for field in required_fields:
            # Extract values for this field from all samples
            values = [feature[field] for feature in features]
            
            # Convert to tensor if not already
            if isinstance(values[0], (list, tuple)):
                values = [torch.tensor(v) for v in values]
            elif not isinstance(values[0], torch.Tensor):
                values = [torch.tensor(v) for v in values]
            
            # Stack tensors along the batch dimension
            batch[field] = torch.stack(values)
            
            # Handle 3D tensors with shape [batch, 1, seq_len]
            if batch[field].dim() == 3 and batch[field].shape[1] == 1:
                batch[field] = batch[field].squeeze(1)
        
        return batch
    
    ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_ds = ds["train"]
    eval_ds = ds["test"]
    
    if config.model_config.use_deepspeed:
        ds_config = config.training_config.stage2.deepspeed_config_file
    else:
        ds_config = None
    
    sft_config = SFTConfig(
        per_device_train_batch_size=config.training_config.stage2.train_batch_size,
        per_device_eval_batch_size=config.training_config.stage2.eval_batch_size,
        gradient_accumulation_steps=config.training_config.stage2.gradient_accumulation_steps,
        num_train_epochs=config.training_config.stage2.num_epochs,
        torch_compile=config.training_config.stage2.torch_compile,
        deepspeed=ds_config if config.model_config.use_deepspeed.use_zero3 else None,
        fp16=config.training_config.stage2.fp16,
        bf16=config.training_config.stage2.bf16,
        gradient_checkpointing=config.training_config.stage2.gradient_checkpointing,
        logging_steps=config.training_config.stage2.logging_steps,
        save_steps=config.training_config.stage2.save_steps,
        save_total_limit=config.training_config.stage2.save_total_limit,
        run_name=f"{config.training_config.stage2.run_name}-{dt}",
        output_dir=f"{config.training_config.stage2.output_dir}",
        save_safetensors=config.training_config.stage2.save_safetensors,
        save_only_model=config.training_config.stage2.save_only_model,
        max_grad_norm=config.training_config.stage2.max_grad_norm,
        max_seq_length=config.training_config.stage2.max_seq_length,
        max_length = config.training_config.stage2.max_length,
        save_strategy=config.training_config.stage2.save_strategy,
        eval_strategy=config.training_config.stage2.eval_strategy,
        report_to=config.training_config.stage2.report_to,
        remove_unused_columns=False,
    )
    print(f"train ds: {train_ds}") if accelerator.is_main_process else None

    trainer = CustomSFTTrainer(
        model=combined_model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
        processing_class=tokenizer,
        data_collator=custom_collate_fn,
    )
    print("args : ", trainer.args) if accelerator.is_main_process else None

    if config.training_stage_config.train_stage2:
        print("Starting training stage 2") if accelerator.is_main_process else None
        if config.training_config.stage2.resume_from_checkpoint:
            print("Resuming training stage 2") if accelerator.is_main_process else None
            trainer.train(resume_from_checkpoint=True)
            print("Training stage 2 completed") if accelerator.is_main_process else None
        else:
            print("Starting training stage 2 from scratch") if accelerator.is_main_process else None
            trainer.train()
            print("Training stage 2 completed") if accelerator.is_main_process else None
        combined_model.save_model(config.training_config.stage2.output_dir)
        print(f"Model saved to {config.training_config.stage2.output_dir}") if accelerator.is_main_process else None
    else:
        print("Training stage 2 not enabled") if accelerator.is_main_process else None

    output_dir = config.training_config.stage2.output_dir
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        print(f"Latest checkpoint found: {latest_checkpoint}")
        combined_model_path = os.path.join(output_dir, latest_checkpoint)
    else:
        print("No checkpoints found in output directory") if accelerator.is_main_process else None
        combined_model_path = output_dir

    return combined_model_path

    
