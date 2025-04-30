from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
import torch.nn as nn
import torch.nn.functional as F
import torch

class TrainingParams:
    def __init__(self, config, model, tokenizer=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

    def setup_optimizer(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["trainer_config"]["learning_rate"],
            betas=(self.config["trainer_config"]["beta1"], self.config["trainer_config"]["beta2"]),
            weight_decay=self.config["trainer_config"]["weight_decay"]
        )
        return optimizer

    def setup_scheduler(self, optimizer, num_training_steps):
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * self.config["trainer_config"]["warmup_ratio"]),
            num_training_steps=num_training_steps
        )
        return scheduler

    def setup_accelerator(self):
        accelerator = Accelerator(
            mixed_precision="fp16" if self.config["trainer_config"]["mixed_precision"] else "no",
        )
        return accelerator
    
    def setup_training(self, train_loader):
        accelerator = self.setup_accelerator()
        optimizer = self.setup_optimizer()
        
        num_training_steps = len(train_loader) * self.config["trainer_config"]["num_epochs"]
        scheduler = self.setup_scheduler(optimizer, num_training_steps)
        
        # Prepare model, optimizer, and dataloaders with accelerator
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            self.model, optimizer, train_loader, scheduler
        )
        
        return model, optimizer, train_loader, scheduler, accelerator