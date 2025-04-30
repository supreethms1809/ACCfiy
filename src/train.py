import torch
import model_arch
import training_params
from train_func import trainFunc
from dataloaders import DataLoaders
from accelerate import Accelerator
from transformers import AutoTokenizer
import yaml
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_config"]["tokenizer_name"], trust_remote_code=True)
    vocab_size = len(tokenizer)
    dim = config["model_config"]["dim"]
    num_layers = config["model_config"]["num_layers"]
    num_heads = config["model_config"]["num_heads"]
    ff_hidden = config["model_config"]["ff_hidden_dim"]

    # Initialize the model
    #accelerator = Accelerator(mixed_precision="fp16")
    model_instance = model_arch.DecoderEncoderDecoderModel(vocab_size, dim=dim, depth=num_layers, heads=num_heads, ff_hidden=ff_hidden).to(device)

    dataloaders = DataLoaders(name=config["dataloader_config"]["dataset_name"],
                              batch_size=config["dataloader_config"]["batch_size"],
                              tokenizer=tokenizer,
                              max_length=config["dataloader_config"]["max_length"],
                              size =config["dataloader_config"]["size"],
                              split=0.9)
    train_loader, val_loader = dataloaders.create_dataloaders()
    #print(f"Train loader size: {len(train_loader)}, Validation loader size: {len(val_loader)}")

    # Initialize training parameters
    training_params_instance = training_params.TrainingParams(config, model_instance, tokenizer)
    model, optimizer, train_loader, scheduler, accelerator = training_params_instance.setup_training(train_loader)
    #print(f"Model: {model}, Optimizer: {optimizer}, Scheduler: {scheduler}")

    # Train the decoder model
    train_func_instance = trainFunc(model, optimizer, train_loader, scheduler, accelerator, tokenizer)
    model, avg_loss, perplexity, losses_per_epoch = train_func_instance.train_decoder_next_tok_pred(config)
    print(f"Average Loss: {avg_loss}, Perplexity: {perplexity}")

    # Train the whole model
    