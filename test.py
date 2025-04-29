# set system path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch 
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer

# Load small sample dataset
dataset_path = './dataset/babeltower'
dataset = load_from_disk(dataset_path)
dataset = dataset.select(range(5))  # smaller for prototype
max_new_tokens = 4098

# Load pre-trained decoder model (DeepSeek R1)
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto", device_map=device)
model.to(device)
print(model)
exit()

# Load encoder model for analysis embeddings
encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
encoder_model = SentenceTransformer(encoder_model_name).to(device)

# Define prompt template
prompt_template = """
Analyze the following C++ code for parallelization opportunities and suggest actions to parallelize it.

Code:
{code}

Output format:
[ANALYSIS]: (your analysis text)
[ACTIONS]: (list of actions to take)
"""

for example in dataset:
    code_str = example['code']

    # Create prompt
    prompt = prompt_template.format(code=code_str)

    # Tokenize and move to device
    input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate output
    outputs = model.generate(
        input_ids=input["input_ids"],
        attention_mask=input["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False,
        output_hidden_states=False
    )

    decoded_output = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    print(f"Model Output:\n{decoded_output}") 

    # Parse output into analysis and actions
    analysis_text = ""
    actions_text = ""
    if "[ANALYSIS]:" in decoded_output and "[ACTIONS]:" in decoded_output:
        try:
            analysis_text = decoded_output.split("[ANALYSIS]:")[1].split("[ACTIONS]:")[0].strip()
            actions_text = decoded_output.split("[ACTIONS]:")[1].strip()
        except Exception as e:
            print(f"Parsing error: {e}")

    # Encode analysis text into embedding
    analysis_embedding = encoder_model.encode(analysis_text, convert_to_tensor=True)
    print(f"Analysis Embedding Shape: {analysis_embedding.shape}")

    # Create a two-layer mapper: analysis_embedding -> action_embedding
    hidden_size = analysis_embedding.shape[-1]
    embed_size = model.model.embed_tokens.embedding_dim  # embed_tokens.embedding_dim

    dummy_mapper = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, embed_size)
    ).to(device)

    action_embedding = dummy_mapper(analysis_embedding)

    print(f"Action Embedding Shape: {action_embedding.shape}")

    # ---------------------------------------------------
    # Now, use a second instance of the decoder to perform transformation
    # ---------------------------------------------------

    # Load second instance (if not already shared weights, can re-use model otherwise)
    # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto", device_map=device)
    # model.to(device)

    # Prepare the transformation prompt
    transform_prompt_template = """
    Given the following C++ code and the action embedding vector, transform the code into an optimized CUDA version by following the actions indicated by the vector.
    The output should not contain any additional text or explanations, just the transformed code.

    Code:
    {code}

    Output format:
    [TRANSFORMED CODE]:
    (your CUDA C++ code here)
    """

    transform_prompt = transform_prompt_template.format(
        code=code_str
    )

    transform_input = tokenizer(transform_prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Expand action embedding across sequence
    input_embeds = model.model.embed_tokens(transform_input["input_ids"])

    action_embeds = action_embedding.unsqueeze(0).unsqueeze(1)  # (1,1,hidden_dim)
    action_embeds = action_embeds.expand(input_embeds.size(0), input_embeds.size(1), -1)  # (batch_size, seq_len, hidden_dim)

    # Add embeddings
    modified_input_embeds = input_embeds + 0.01 * action_embeds  # Small scaling to avoid destabilizing

    # ---------------------------------------------------
    # Generate transformed CUDA code using prompt
    # ---------------------------------------------------
    transform_outputs = model.generate(
        inputs_embeds=modified_input_embeds,
        attention_mask=transform_input["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False
    )

    transformed_code = tokenizer.batch_decode(transform_outputs.sequences, skip_special_tokens=True)[0]
    print("-"*50)
    print(f"Transformed CUDA Code:\n{transformed_code}")
