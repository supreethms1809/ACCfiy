import unsloth
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from unsloth import FastLanguageModel as unsloth_model
from src.model.qwen_combined_model import QwenCombinedModel
import torch
import os

def init_model_and_tokenizer_normal(config, decoder1, decoder2, tokenizer, model_config):
    print("Initializing model and tokenizer")
    if config.model_config.attn_implementation == "flash_attention_2":
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"

    if config.model_config.use_dtype == "fp16":
        dtype = torch.float16
    elif config.model_config.use_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, dtype=dtype)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<task>", "</task>", "<code_cpp>", "</code_cpp>", "<analysis>", "</analysis>", "<code_cuda>", "</code_cuda>", "<kernel>", "</kernel>", "<think>", "</think>"
            ]
        })
    else:
        tokenizer = tokenizer
    if decoder1 is None:
        decoder1, _ = AutoModelForCausalLM.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, attn_implementation=attn_implementation, dtype=dtype)
        decoder1.resize_token_embeddings(len(tokenizer))
    else:
        decoder1 = decoder1
    if decoder2 is None:
        decoder2, _ = AutoModelForCausalLM.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, attn_implementation=attn_implementation, dtype=dtype)
        decoder2.resize_token_embeddings(len(tokenizer))
    else:
        decoder2 = decoder2
    if model_config is None:
        model_config = AutoConfig.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, dtype=dtype)
    else:
        model_config = model_config
    return decoder1, decoder2, tokenizer, model_config

def init_model_and_tokenizer_unsloth(config, decoder1, decoder2, tokenizer, model_config):
    print("Initializing model and tokenizer")
    if config.model_config.attn_implementation == "flash_attention_2":
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"

    if config.model_config.use_dtype == "fp16":
        dtype = torch.float16
    elif config.model_config.use_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, dtype=dtype, device_map=None)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<task>", "</task>", "<code_cpp>", "</code_cpp>", "<analysis>", "</analysis>", "<code_cuda>", "</code_cuda>", "<kernel>", "</kernel>", "<think>", "</think>"
            ]
        })
    else:
        tokenizer = tokenizer
    if decoder1 is None:
        decoder1, _ = unsloth_model.from_pretrained(config.base_model.base_model_hf_name, 
                                                trust_remote_code=True, 
                                                attn_implementation=attn_implementation, 
                                                dtype=dtype, 
                                                device_map=None, 
                                                full_finetuning = True,
                                                load_in_8bit = False,
                                                load_in_4bit = False,
                                                )
        decoder1.resize_token_embeddings(len(tokenizer))
    else:
        decoder1 = decoder1
    if decoder2 is None:
        decoder2, _ = unsloth_model.from_pretrained(config.base_model.base_model_hf_name, 
                                                trust_remote_code=True, 
                                                attn_implementation=attn_implementation, 
                                                dtype=dtype, 
                                                device_map=None, 
                                                full_finetuning = True,
                                                load_in_8bit = False,
                                                load_in_4bit = False,
                                                )
        decoder2.resize_token_embeddings(len(tokenizer))
    else:
        decoder2 = decoder2
    if model_config is None:
        model_config = AutoConfig.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, dtype=dtype, device_map=None)
    else:
        model_config = model_config
    return decoder1, decoder2, tokenizer, model_config

def init_combined_model(config, decoder1, decoder2, tokenizer, model_config, mapper_state, init=True):
    if init:
        if config.model_config.use_unsloth:
            decoder1, decoder2, tokenizer, model_config = init_model_and_tokenizer_unsloth(config, decoder1, decoder2, tokenizer, model_config)
        else:
            decoder1, decoder2, tokenizer, model_config = init_model_and_tokenizer_normal(config, decoder1, decoder2, tokenizer, model_config)
    combined_model = QwenCombinedModel(config, decoder1, decoder2, tokenizer, model_config, mapper_state=mapper_state)
    return combined_model

def load_combined_model(config, decoder1_path=None, decoder2_path=None, mapper_path=None):
    if config.model_config.attn_implementation == "flash_attention_2":
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"
    if config.model_config.use_dtype == "fp16":
        dtype = torch.float16
    elif config.model_config.use_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    try:
        if decoder1_path is not None:
            print("Loading decoder1 from: ", decoder1_path)
            if config.model_config.use_unsloth:
                decoder1, _ = unsloth_model.from_pretrained(decoder1_path, trust_remote_code=True, attn_implementation=attn_implementation, dtype=dtype, device_map=None, full_finetuning = True, load_in_8bit = False, load_in_4bit = False)
            else:
                decoder1, _ = AutoModelForCausalLM.from_pretrained(decoder1_path, trust_remote_code=True, attn_implementation=attn_implementation, dtype=dtype)
            tokenizer = AutoTokenizer.from_pretrained(decoder1_path, trust_remote_code=True, dtype=dtype)
            model_config = AutoConfig.from_pretrained(decoder1_path, trust_remote_code=True, dtype=dtype)
        else:
            decoder1 = None
    except Exception as e:
        print(f"Error loading combined model: {e}, Using the base decoder1 model")
    
    try:
        if decoder2_path is not None:
            print("Loading decoder2 from: ", decoder2_path)
            if config.model_config.use_unsloth:
                decoder2, _ = unsloth_model.from_pretrained(decoder2_path, trust_remote_code=True, attn_implementation=attn_implementation, dtype=dtype, device_map=None, full_finetuning = True, load_in_8bit = False, load_in_4bit = False)
            else:
                decoder2, _ = AutoModelForCausalLM.from_pretrained(decoder2_path, trust_remote_code=True, attn_implementation=attn_implementation, dtype=dtype)
            tokenizer = AutoTokenizer.from_pretrained(decoder2_path, trust_remote_code=True, dtype=dtype)
            model_config = AutoConfig.from_pretrained(decoder2_path, trust_remote_code=True, dtype=dtype)
        else:
            decoder2 = None
    except Exception as e:
        print(f"Error loading combined model: {e}, Using the base decoder2 model")

    try:
        if mapper_path is not None:
            print("Loading mapper from: ", mapper_path)
            mapper_state = torch.load(mapper_path)
            print("Mapper state loaded successfully")
        else:
            mapper_state = None
    except Exception as e:
        print(f"Error loading combined model: {e}, Using the base mapper")

    if decoder1_path is not None and decoder2_path is not None and mapper_path is not None:
        init = False
    else:
        init = True
    
    combined_model = init_combined_model(config, decoder1, decoder2, tokenizer, model_config, mapper_state, init)
    print("Successfully loaded combined model")

    return combined_model, tokenizer, model_config

def load_decoder1(config):
    if config.model_config.attn_implementation == "flash_attention_2":
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"
    if config.model_config.use_dtype == "fp16":
        dtype = torch.float16
    elif config.model_config.use_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        
    if config.model_config.use_unsloth:
        decoder1, _ = unsloth_model.from_pretrained(config.base_model.base_model_hf_name, 
                                                 trust_remote_code=True, 
                                                 dtype=dtype, 
                                                 attn_implementation=attn_implementation, 
                                                 device_map=None, 
                                                 full_finetuning = True,
                                                 load_in_8bit = False,
                                                 load_in_4bit = False,
                                                 )
        model_config = AutoConfig.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<task>", "</task>", "<code_cpp>", "</code_cpp>", "<analysis>", "</analysis>", "<code_cuda>", "</code_cuda>", "<kernel>", "</kernel>", "<think>", "</think>"
            ]
        })
        tokenizer.padding_side = "left"
        decoder1.resize_token_embeddings(len(tokenizer))
    else:
        decoder1, _ = AutoModelForCausalLM.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, attn_implementation=attn_implementation, dtype=dtype)
        model_config = AutoConfig.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<task>", "</task>", "<code_cpp>", "</code_cpp>", "<analysis>", "</analysis>", "<code_cuda>", "</code_cuda>", "<kernel>", "</kernel>", "<think>", "</think>"
            ]
        })
        tokenizer.padding_side = "left"
        decoder1.resize_token_embeddings(len(tokenizer))
    return decoder1, model_config, tokenizer

def get_decoder1_path(config):
    output_dir = config.training_config.stage1.output_dir
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        print(f"Latest checkpoint found: {latest_checkpoint}")
        decoder1_path = os.path.join(output_dir, latest_checkpoint)
    else:
        print("No checkpoints found in output directory")
        decoder1_path = output_dir

    return decoder1_path