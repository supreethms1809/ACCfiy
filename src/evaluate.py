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
                              split=0.9,
                              num_workers=config["dataloader_config"]["num_workers"])
    train_loader, val_loader = dataloaders.create_dataloaders()
    #print(f"Train loader size: {len(train_loader)}, Validation loader size: {len(val_loader)}")

    # Initialize training parameters
    training_params_instance = training_params.TrainingParams(config, model_instance, tokenizer)
    model, optimizer, train_loader, scheduler, accelerator = training_params_instance.setup_training(train_loader)
    #print(f"Model: {model}, Optimizer: {optimizer}, Scheduler: {scheduler}")

    # Train the decoder model
    # Load the saved checkpoint
    train_func_instance = trainFunc(model, optimizer, train_loader, scheduler, accelerator, tokenizer)
    model, optimizer, scheduler, epoch = train_func_instance.load_checkpoint(config["trainer_config"]["checkpoint_path"])
    test_prompt = "Overflow pipes at the Burry Inlet near Llanelli are used"
    input_ids = tokenizer(test_prompt, return_tensors="pt")["input_ids"].to(accelerator.device)
    with torch.no_grad():
        generated_ids = model.generate_decoderonly(input_ids, max_new_tokens=512)
        print("Generated text:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))
