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
input_str = "I want to convert a code from cpp to "
input = tokenizer(input_str, return_tensors="pt", padding =True, truncation=True).to("cuda:0")
#print(f"model: {model}")
#print(f"input: {input.keys()}")
output = model.generate(
    input_ids=input["input_ids"],
    attention_mask=input["attention_mask"],
    max_new_tokens=128,
    # return_dict_in_generate=True,
    # output_hidden_states=True,
    pad_token_id=tokenizer.eos_token_id,
)
output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
print(f"Output: {output}")
# all_tokens_last_layer = torch.cat(
#     [token_hidden_states[-1] for token_hidden_states in output.hidden_states],
#     dim=1
# )
# print(all_tokens_last_layer.shape)

# pass the hidden states to the encoder model
encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
from sentence_transformers import SentenceTransformer
encoder_model = SentenceTransformer(encoder_model_name).to("cuda:0")
#encoder_input = encoder_tokenizer(output, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
analysis_embedding = encoder_model.encode(output, convert_to_tensor=True)
print(f"analysis embedding: {analysis_embedding.shape}")
# for example in dataset:
#     code_str = example['code']


