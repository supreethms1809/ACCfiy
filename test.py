# set system path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create a dataset class inside source and process this
dataset = './dataset/babeltower'
from datasets import load_from_disk
dataset = load_from_disk(dataset)
dataset = dataset.select(range(20))
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# create model inference class for this
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto", device_map="cuda:0")
model.to("cuda:0")
input_str = "Write a python function to calculate the factorial of a number."
input = tokenizer(input_str, return_tensors="pt", padding =True, truncation=True).to("cuda:0")
#print(f"model: {model}")
#print(f"input: {input.keys()}")
output = model.generate(
    input_ids=input["input_ids"],
    attention_mask=input["attention_mask"],
    max_new_tokens=128,
    return_dict_in_generate=True,
    output_hidden_states=True,
    pad_token_id=tokenizer.eos_token_id,
)

all_tokens_last_layer = torch.cat(
    [token_hidden_states[-1] for token_hidden_states in output.hidden_states],
    dim=1
)
print(all_tokens_last_layer.shape)

# pass the hidden states to the encoder model
encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
from sentence_transformers import SentenceTransformer
encoder_model = SentenceTransformer(encoder_model_name)
print(f"encoder model: {encoder_model}")

# for example in dataset:
#     code_str = example['code']


