from datasets import load_dataset
import torch

class accfiyDataset:
    def __init__(self, config, tokenizer):
        self.config = config
        self.dataset_name = config.training_config.stage1.dataset_name
        self.hf_dataset_path = config.training_config.stage1.hf_dataset_path
        self.split = config.training_config.stage1.split
        self.max_length = config.training_config.stage1.max_length
        self.test_run = config.training_config.stage1.test_run
        self.test_run_size = config.training_config.stage1.test_run_size
        self.tokenizer = tokenizer

    def load_dataset(self):
        ds = load_dataset(self.hf_dataset_path, split=self.split)
        if self.test_run:
            ds = ds.select(range(self.test_run_size))
        return ds

    def tokenize_dataset(self, ds):
        def tokenize_function(example):
            messages = [
                {
                    "role": "user",
                    "content": f"You are a helpful assistant that analyzes code for GPU parallelization. \
                        Your task is to analyze C++ code and provide an analysis and instructions for parallelization of the code using CUDA. \
                        <task> ANALYZE_FOR_PARALLELIZATION </task>\n<code_cpp>\n{example['cpp_code']}\n</code_cpp>"
                },
                {
                    "role": "assistant",
                    "content": f"<analysis>\n{example['analysis']}\n</analysis>\n<instructions>\n{example['instructions']}\n</instructions>"
                }
            ]
            
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            tokenized = self.tokenizer(
                full_prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            
            labels = input_ids.clone()
            labels[:-1].fill_(-100)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        tokenized_ds = ds.map(
            tokenize_function,
            remove_columns=ds.column_names,
            num_proc=32
        )
        #print(tokenized_ds[0])
        size_check = self.check_batch_size(tokenized_ds)
        if size_check:
            print("Batch size check passed")

        return tokenized_ds
    
    def check_batch_size(self, ds):
        try:
            first_entry = ds[0]
            expected_lengths = {k: len(v) for k, v in first_entry.items()}
            for i in range(1, len(ds)):
                entry = ds[i]
                for k, v in entry.items():
                    if len(v) != expected_lengths[k]:
                        print(f"Entry {i} has different length for {k}: expected {expected_lengths[k]}, got {len(v)}")
                        return False
            return True
        except Exception as e:
            print(f"Error checking batch sizes: {str(e)}")
            return False
        
class accfiySyntheticDataset:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_name = config.training_config.stage2.dataset_name
        self.hf_dataset_path = config.training_config.stage2.hf_dataset_path
        self.split = config.training_config.stage2.split
        self.max_length = config.training_config.stage2.max_length
        self.test_run = config.training_config.stage2.test_run
        self.test_run_size = config.training_config.stage2.test_run_size
        
    def load_dataset(self):
        ds = load_dataset(self.hf_dataset_path, split=self.split)
        if self.test_run:
            ds = ds.select(range(self.test_run_size))
        return ds
    
    def tokenize_dataset(self, ds):
        def tokenize_function(example):
            messages = [
                {
                    "role": "user",
                    "content": f"You are a helpful assistant that transforms the CPU code to GPU code. \
                        Your task is to Generate the CUDA kernel function for the corresponding C++ code. \
                        <task> TRANSFORM_C++_TO_CUDA_KERNEL </task>\n<code_cpp>\n{example['cpp_code']}\n</code_cpp>"
                },
                {
                    "role": "assistant",
                    "content": f"<kernel>\n{example['cuda_code']}\n</kernel>"
                }
            ]
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            tokenized = self.tokenizer(
                full_prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            
            labels = input_ids.clone()
            labels[:-1].fill_(-100)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        tokenized_ds = ds.map(
            tokenize_function,
            remove_columns=ds.column_names,
            num_proc=32
        )
        size_check = self.check_batch_size(tokenized_ds)
        if size_check:
            print("Batch size check passed")

        return tokenized_ds
    
    def check_batch_size(self, ds):
        try:
            first_entry = ds[0]
            expected_lengths = {k: len(v) for k, v in first_entry.items()}
            for i in range(1, len(ds)):
                entry = ds[i]
                for k, v in entry.items():
                    if len(v) != expected_lengths[k]:
                        print(f"Entry {i} has different length for {k}: expected {expected_lengths[k]}, got {len(v)}")
                        return False
            return True
        except Exception as e:
            print(f"Error checking batch sizes: {str(e)}")
            return False