import unsloth
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from unsloth import FastLanguageModel as unsloth_model
from src.model.qwen_combined_model import QwenCombinedModel
import torch
import os

def init_model_and_tokenizer_normal(config):
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
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, dtype=dtype)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<task>", "</task>", "<code_cpp>", "</code_cpp>", "<analysis>", "</analysis>", "<code_cuda>", "</code_cuda>", "<kernel>", "</kernel>", "<think>", "</think>"
        ]
    })
    decoder1, _ = AutoModelForCausalLM.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, attn_implementation=attn_implementation, dtype=dtype)
    decoder2, _ = AutoModelForCausalLM.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, attn_implementation=attn_implementation, dtype=dtype)
    decoder1.resize_token_embeddings(len(tokenizer))
    decoder2.resize_token_embeddings(len(tokenizer))
    model_config = AutoConfig.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, dtype=dtype)
    return decoder1, decoder2, tokenizer, model_config

def init_model_and_tokenizer_unsloth(config):
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
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, dtype=dtype, device_map=None)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<task>", "</task>", "<code_cpp>", "</code_cpp>", "<analysis>", "</analysis>", "<code_cuda>", "</code_cuda>", "<kernel>", "</kernel>", "<think>", "</think>"
        ]
    })
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
    model_config = AutoConfig.from_pretrained(config.base_model.base_model_hf_name, trust_remote_code=True, dtype=dtype, device_map=None)
    return decoder1, decoder2, tokenizer, model_config

def init_combined_model(config, decoder1, decoder2, tokenizer, model_config, mapper_state, init=True):
    if init:
        if config.model_config.use_unsloth:
            decoder1, decoder2, tokenizer, model_config = init_model_and_tokenizer_unsloth(config)
        else:
            decoder1, decoder2, tokenizer, model_config = init_model_and_tokenizer_normal(config)
    combined_model = QwenCombinedModel(config, decoder1, decoder2, tokenizer, model_config, mapper_state=mapper_state)
    return combined_model

def load_combined_model(config, decoder1_path=None, decoder2_path=None, mapper_path=None):
    try:
        if os.path.exists(config.model_config.combined_model_decoder1_local_path) or decoder1_path is not None:
            decoder1 = AutoModelForCausalLM.from_pretrained(config.model_config.combined_model_decoder1_local_path)
            tokenizer = AutoTokenizer.from_pretrained(config.model_config.combined_model_decoder1_local_path)
            model_config = AutoConfig.from_pretrained(config.model_config.combined_model_decoder1_local_path)
    except Exception as e:
        print(f"Error loading combined model: {e}, Using the base decoder1 model")
        mapper_state = None
    
    try:
        if os.path.exists(config.model_config.combined_model_decoder2_local_path) or decoder2_path is not None:
            decoder2 = AutoModelForCausalLM.from_pretrained(config.model_config.combined_model_decoder2_local_path)
            tokenizer = AutoTokenizer.from_pretrained(config.model_config.combined_model_decoder2_local_path)
            model_config = AutoConfig.from_pretrained(config.model_config.combined_model_decoder2_local_path)
    except Exception as e:
        print(f"Error loading combined model: {e}, Using the base decoder2 model")
        mapper_state = None
    
    try:
        if os.path.exists(config.model_config.combined_model_mapper_local_path) or mapper_path is not None:
            mapper_state = torch.load(config.model_config.combined_model_mapper_local_path)
    except Exception as e:
        print(f"Error loading combined model: {e}, Using the base mapper")
        mapper_state = None

    if decoder1_path is not None and decoder2_path is not None and mapper_path is not None:
        init = False
    else:
        init = True
        decoder1 = None
        decoder2 = None
        mapper_state = None
        tokenizer = None
        model_config = None
        
    combined_model = init_combined_model(config, decoder1, decoder2, tokenizer, model_config, mapper_state, init)
    print("Successfully loaded combined model")

    return combined_model

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
        decoder1.resize_token_embeddings(len(tokenizer))
    return decoder1, model_config, tokenizer