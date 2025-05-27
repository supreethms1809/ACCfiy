import torch
import torch.nn as nn

class QwenCombinedModel:
    def __init__(self, config, decoder1, decoder2, tokenizer, model_config=None, mapper_state=None):
        self.config = config
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.tokenizer = tokenizer
        self.model_config = model_config
        if mapper_state is None:
            self.mapper = nn.Sequential(
                nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size)
            )
        else:
            self.mapper = nn.Sequential(
                nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size)
            )
            self.mapper.load_state_dict(mapper_state)

    def forward(self, input_ids, attention_mask):
        pass