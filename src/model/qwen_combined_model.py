import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin

class QwenCombinedModel(PreTrainedModel, GenerationMixin):
    def __init__(self, config, decoder1, decoder2, tokenizer, model_config=None, mapper_state=None):
        super().__init__(model_config)
        self.run_config = config
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.tokenizer = tokenizer
        self.config = model_config
        if self.run_config.model_config.use_dtype == "fp16":
            self.config.torch_dtype = "torch.float16"
        elif self.run_config.model_config.use_dtype == "bf16":
            self.config.torch_dtype = "torch.bfloat16"
        else:
            self.config.torch_dtype = "torch.float32"
        if mapper_state is None:
            self.mapper = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size)
            )
        else:
            self.mapper = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size)
            )
            self.mapper.load_state_dict(mapper_state)
        self.decoder1.resize_token_embeddings(len(self.tokenizer))
        self.decoder2.resize_token_embeddings(len(self.tokenizer))
        self.config.use_cache = False
        self.config.attn_implementation = "flash_attention_2"
        self.mapper_gradient_checkpointing = False
        self.warnings_issued = {}
        self.add_model_tags = lambda *args, **kwargs: None
        self.max_position_embeddings = self.config.max_position_embeddings

    def forward(self, input_ids, attention_mask, labels, input_ids_decoder1, **kwargs):
        print(f"input_ids.shape: {type(input_ids)}")
        print(f"attention_mask.shape: {type(attention_mask)}")
        print(f"labels.shape: {type(labels)}")
        print(f"input_ids_decoder1.shape: {type(input_ids_decoder1)}")

        # Clone and detach inputs to ensure they're in the correct state
        input_ids = input_ids.clone().detach()
        attention_mask = attention_mask.clone().detach()
        labels = labels.clone().detach()
        input_ids_decoder1 = input_ids_decoder1.clone().detach()

        # Generate with decoder1 in inference mode
        with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
            decoder1_output = self.decoder1.generate(
                input_ids=input_ids_decoder1,  
                max_new_tokens=self.run_config.training_config.stage2.max_new_tokens,
                temperature=self.run_config.combined_model.combined_model_decoder1_temperature,
                top_p=self.run_config.combined_model.combined_model_decoder1_top_p,
                top_k=self.run_config.combined_model.combined_model_decoder1_top_k,
                min_p=self.run_config.combined_model.combined_model_decoder1_min_p,
                use_cache=self.run_config.combined_model.combined_model_use_cache,
            )
        
        # Clone and detach decoder1 output
        decoder1_output = decoder1_output.clone().detach()
        
        # Get embeddings and ensure they're in the correct state
        decoder1_embeddings = self.decoder1.get_input_embeddings()(decoder1_output)
        decoder1_embeddings = decoder1_embeddings.clone().detach()
        
        # Process through mapper with autocast
        with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
            mapper_embeddings = self.mapper(decoder1_embeddings)
            # Use addition instead of inplace add
            modulated_embeddings = mapper_embeddings + decoder1_embeddings
        
        # Enable gradients for the modulated embeddings
        modulated_embeddings.requires_grad_(True)

        # Generate with decoder2
        with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
            outputs2 = self.decoder2(
                input_ids=input_ids,
                context=modulated_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        return outputs2

    def get_input_embeddings(self):
        return self.decoder1.get_input_embeddings()
    
    def get_output_embeddings(self):
        return self.decoder2.get_output_embeddings()
    
    def get_hidden_size(self):
        return self.config.hidden_size
    
    def enable_mapper_gradient_checkpointing(self):
        self.mapper_gradient_checkpointing = True

    def disable_mapper_gradient_checkpointing(self):
        self.mapper_gradient_checkpointing = False

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.decoder1, "gradient_checkpointing_enable"):
            self.decoder1.gradient_checkpointing_enable(**kwargs)
        if hasattr(self.decoder2, "gradient_checkpointing_enable"):
            self.enable_mapper_gradient_checkpointing = True
            self.decoder2.gradient_checkpointing_enable(**kwargs)
        self.config.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.decoder1, "gradient_checkpointing_disable"):
            self.decoder1.gradient_checkpointing_disable()
        if hasattr(self.decoder2, "gradient_checkpointing_disable"):
            self.enable_mapper_gradient_checkpointing = False
            self.decoder2.gradient_checkpointing_disable()
        self.config.gradient_checkpointing = False

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for the generate method, handling different generation strategies."""
        attention_mask = kwargs.get("attention_mask", None)
        
        # Initial step: get hidden states from decoder1
        if past_key_values is None:
            # First step - standard preparation
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "use_cache": kwargs.get("use_cache", False),
            }
        else:
            # Subsequent steps - use cached computation
            return {
                "input_ids": input_ids[:, -1:],
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", False),
            }

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search."""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) 
                                    for past_state in layer_past),)
        return reordered_past
    