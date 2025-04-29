import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from src.datasets import BilingualDataset, casual_mask

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


### TOKENIZER UTILS ###
def get_all_sentences(ds, lang):
    for example in ds:
        yield example["translation"][lang]

def get_or_build_tokenizer(opt, ds, lang): # lang is the language of the dataset, ds is the dataset
    tokenizer_path = Path(opt.input.tokenizer_path.format(lang))
    print(tokenizer_path)
    if not tokenizer_path.exists():
        print("Building tokenizer")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace() # Split on whitespace
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # for a word to appear in the vocab is has to appear at least twice
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


### MODEL UTILS ###

import numpy as np
import random
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from datasets import load_dataset

from datetime import timedelta

import wandb

from src.model import Transformer

def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt


def get_model_and_optimizer(opt):
    model = Transformer(opt)
    # model = FFCCVAE_model_classifiers.FFCCVAE(opt)

    if "cuda" in opt.device or "mps" in opt.device:
        model = model.to(opt.device)
    print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.loss_fn.parameters())
    ]

    if opt.training.optimizer == "SGD":
        
        optimizer = torch.optim.SGD(
            [
                {
                    "params": main_model_params,
                    "lr": opt.training.learning_rate,
                    "weight_decay": opt.training.weight_decay,
                    "momentum": opt.training.momentum,
                },
                {
                    "params": model.loss_fn.parameters(),
                    "lr": opt.training.downstream_learning_rate,
                    "weight_decay": opt.training.downstream_weight_decay,
                    "momentum": opt.training.momentum,
                },
            ]
        )
    elif opt.training.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [
                {
                    "params": main_model_params,
                    "lr": opt.training.learning_rate,
                    "weight_decay": opt.training.weight_decay,
                    "betas": (opt.training.betas[0] , opt.training.betas[1]), 
                },
                {
                    "params": model.loss_fn.parameters(),
                    "lr": opt.training.downstream_learning_rate,
                    "weight_decay": opt.training.downstream_weight_decay,
                    "betas": (opt.training.betas[0] , opt.training.betas[1]), 
                },
            ]
        )
    return model, optimizer



def get_data_or_tokenizer(opt, partition, data=True):
    if opt.input.dataset == "opus_books":
        ds_raw = load_dataset("opus_books", f"{opt.input.lang_src}-{opt.input.lang_tgt}", split="train")
        
    else:
        raise ValueError("Unknown dataset.")
    

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(opt, ds_raw, opt.input.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(opt, ds_raw, opt.input.lang_tgt)

    # Keep 90% for training and 10% for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    ds_train_raw, ds_val_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) #random split splits the dataset randomly into two datasets

    # Build the datasets
    if opt.input.dataset == "opus_books":
        if partition == "train":
            dataset = BilingualDataset(opt, ds_train_raw, tokenizer_src, tokenizer_tgt)
        elif partition == "val" or partition == "test":
            dataset = BilingualDataset(opt, ds_val_raw, tokenizer_src, tokenizer_tgt)
        else:
            raise ValueError("Unknown partition.")
    else:
        raise ValueError("Unknown dataset.")
    
    if opt.input.check_len:
        max_len_src = 0
        max_len_tgt = 0
        for example in ds_raw:
            src_ids = tokenizer_src.encode(example["translation"][opt.input.lang_src]).ids
            tgt_ids = tokenizer_tgt.encode(example["translation"][opt.input.lang_tgt]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f"Max length src: {max_len_src}, max length tgt: {max_len_tgt}")

    

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    if data and partition == "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.input.batch_size,
            drop_last=True,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=7,
            persistent_workers=True
        )
    elif data and partition != "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            drop_last=True,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=7,
            persistent_workers=True
        )
    else:
        return tokenizer_src, tokenizer_tgt


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# This functions are used to send the data to the right device before training.
def preprocess_inputs(opt, inputs):
    if "cuda" in opt.device or "mps" in opt.device:
        inputs = dict_to_cuda(opt, inputs)
        # labels = dict_to_cuda(opt, labels)
    return inputs

def dict_to_cuda(opt, obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key != "src_text" and key != "tgt_text":
                obj[key] = value.to(opt.device, non_blocking=True)
    else:
        obj = obj.to(opt.device, non_blocking=True)
    return obj


# Print results and log them to wandb
def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()
    partition_scalar_outputs = {}
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            partition_scalar_outputs[f"{partition}_{key}"] = value
    wandb.log(partition_scalar_outputs, step=epoch)

# create save_model function
def save_model(model):
    torch.save(model.state_dict(), f"{wandb.run.name}-model.pt")
    # log model to wandb
    wandb.save(f"{wandb.run.name}-model.pt")


# create log_results function
def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict

# cools down after the first half of the epochs
def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr

def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer
