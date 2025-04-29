### SEACH FOR "..." on every file to see where you need to add your code ###


import time
from collections import defaultdict
import os

import hydra
import torch
from omegaconf import DictConfig

from src import utils
import wandb


def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data_or_tokenizer(opt, "train")
    num_steps_per_epoch = len(train_loader)
    print("num_steps_per_epoch:",num_steps_per_epoch)
    best_val_acc = 0.0
    model.train()

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        for batch in train_loader:
            batch = utils.preprocess_inputs(opt, batch) # push to GPU

            # print("input shape:",inputs['sample'].shape)
            # print("label shape:",labels['class_labels'].shape)
            optimizer.zero_grad()

            scalar_outputs = model(batch)
            scalar_outputs["Loss"].backward()

            optimizer.step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            best_val_acc = validate_or_test(opt, model, "val", epoch=epoch, best_val_acc=best_val_acc)

        

    return model


def validate_or_test(opt, model, partition, epoch=None, best_val_acc=1.0, max_visualizations=1):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data_or_tokenizer(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(partition)

    # Counter for visualizations
    visualization_count = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = utils.preprocess_inputs(opt, batch) # push to GPU

            # Enable visualization only if the count is below the threshold
            visualize = visualization_count < max_visualizations
            scalar_outputs = model.predict(
                batch, visualize=visualize
            )
            if visualize:
                visualization_count += 1

            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

            

        utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)
        
        # Save model if classification accuracy is better than previous best
        # if test_results["classification_accuracy"] > best_val_acc :
        #     print("saving model")
        #     best_val_acc = test_results["classification_accuracy"]

def set_run_name(opt):
    
    run_name = f"dmodel:{opt.model.d_model}_{opt.input.lang_src}_{opt.input.lang_tgt}"
    return run_name

@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    # Parse arguments and initialize the Wandb run
    opt = utils.parse_args(opt)
    run = wandb.init(
        project="Transformer", # Wandb project name
        name=set_run_name(opt),  # Wandb creates random run names if you skip this field
        reinit=False,  # Allows reinitializing runs when you re-run this cell
        config=dict(opt)  # Wandb Config for your run
    )

    # Get the directory of the main script (the file where this code is running)
    main_file_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
    model_dir = os.path.join(main_file_dir, "models")  # Save model in the 'models' subdirectory
    model_name = f"{set_run_name(opt)}.pt"
    model_path = os.path.join(model_dir, model_name)
    print(main_file_dir, model_dir, model_name, model_path)

    # Check if the model exists and decide whether to overwrite or load it
    if os.path.exists(model_path):
        if opt.overwrite:  # Check if overwrite is enabled
            print(f"Model exists, but overwriting because 'overwrite' is set to True.")
            model, optimizer = utils.get_model_and_optimizer(opt)
            model = train(opt, model, optimizer)  # Retrain the model if overwrite is True
        else:
            print(f"Loading the existing model from {model_path}.")
            model, optimizer = utils.get_model_and_optimizer(opt)
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set the model to evaluation mode
            print("Model loaded. Skipping training and starting testing.")
    else:
        print(f"Training the model, as no pre-existing model was found.")
        model, optimizer = utils.get_model_and_optimizer(opt)
        model = train(opt, model, optimizer)

    # Run the test if required
    if opt.training.final_test:
        validate_or_test(opt, model, "test")

    # Save the model after training (or after loading if you run testing)
    if (not os.path.exists(model_path) or opt.overwrite) and opt.save:  # Save only if the model was newly trained or overwritten
        torch.save(model.state_dict(), model_path)

    run.finish()

    


if __name__ == "__main__":
    my_main()
