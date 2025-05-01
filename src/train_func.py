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

    def train_decoder_next_tok_pred(self, config, load_checkpoint=False):
        if load_checkpoint:
            trainFuncDecoder = trainFuncDecoder(self.model, self.optimizer, self.scheduler, self.accelerator, self.tokenizer)
            self.model, self.optimizer, self.scheduler, start_epoch, global_step = trainFuncDecoder.load_checkpoint_decoder(config["trainer_config"]["checkpoint_path"])
        else:
            start_epoch = 0
            global_step = 0
        epochs = config["trainer_config"]["num_epochs"]
        losses_per_epoch = []
        best_loss = float("inf")
        #global_step = 0
        save_steps = 10000
        for epoch in tqdm(range(start_epoch, epochs)):
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

                global_step += 1
                # Save checkpoint every save_steps
                if global_step % save_steps == 0:
                    checkpoint = {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step
                    }
                    checkpoint_path = f"checkpoint_step_{global_step}.pt"
                    self.accelerator.save(checkpoint, checkpoint_path)
                    print(f"Checkpoint saved at step {global_step} to {checkpoint_path}")

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
                self.accelerator.save(checkpoint, "checkpoint_best.pt")
                print("Checkpoint saved.")

        return self.model, avg_loss, perplexity, losses_per_epoch
    
class trainFuncDecoder:
    def __init__(self, model, optimizer, scheduler, accelerator, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
    
    def load_checkpoint_decoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
        #checkpoint = torch.load(checkpoint_path)
        model = self.model.decoder.load_state_dict(checkpoint["model_state_dict"])
        optimizer = self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {start_epoch}.")
        return model, optimizer, scheduler ,start_epoch, global_step