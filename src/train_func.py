from tqdm import tqdm
import torch
from torch import nn
import math
from accelerate import Accelerator


class trainFunc:
    def __init__(self, model, optimizer, train_loader, scheduler, accelerator, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.loss_fn = self.loss_fn()

    def loss_fn(self):
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        return criterion

    def train_decoder_next_tok_pred(self, config):
        epochs = config["trainer_config"]["num_epochs"]
        losses_per_epoch = []
        best_loss = float("inf")
        for epoch in tqdm(range(epochs)):
            # Freeze encoder parameters
            self.model.encoder.eval()
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            
            self.model.train()
            total_loss = 0.0
            total_tokens = 0
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for step, (input_ids, target_ids) in enumerate(progress):
                input_ids = input_ids.to(self.accelerator.device)
                target_ids = target_ids.to(self.accelerator.device)
                x = self.model.token_embedding(input_ids)
                mask = self.model._causal_mask(x)
                hidden = self.model.decoder(x, mask=mask)
                logits = self.model.output_proj(hidden)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_tokens += target_ids.numel()
                progress.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.train_loader)
            perplexity = math.exp(total_loss / total_tokens)
            losses_per_epoch.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "epoch": epoch
                }
                self.accelerator.save(checkpoint, "checkpoint.pt")
                print("Checkpoint saved.")

        return self.model, avg_loss, perplexity, losses_per_epoch