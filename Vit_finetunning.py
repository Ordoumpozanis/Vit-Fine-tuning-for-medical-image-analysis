#!/usr/bin/env python
# coding: utf-8

# Ensure necessary installations
# !pip install datasets transformers accelerate torch scikit-learn matplotlib tensorboard codecarbon TensorFlow pillow==9.5.0

#if you want to see the results use the command 
# tensorboard --logdir /workspace/experiment/name_of_experiment_folder --host 0.0.0.0 --port 6006

#HOW TO RUN This CODE
# accelerate lauch filename.py


import os
import time
import csv 
import random
import numpy as np
from datasets import load_dataset
from pathlib import Path
import tempfile
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Resize,
)
from codecarbon import EmissionsTracker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_recall_fscore_support, accuracy_score
from transformers import EvalPrediction

################## SETINGS OF THE CODE ################

do_train = True #if true it will run the training part. Have to be tru in the fine-tunning process
do_eval = True #if true it will run the training part. Have to be tru in the fine-tunning process
do_predict = False #if true it will run the testing part . onl when you have finisted the optimization and you want to use the test dataset

name_of_experiment = "E1.learning_rate"  # On each experiment creates a a folder. this is the name of the Folder

# experiments to make
learning_rates = [1e-1, 1e-3, 1e-5, 1e-6] # This is the an example of a hyperparameter optimization.  in this example 4 values of learning rate will be evaluated
validation_dataset_percentage = 0.1 # In the dataset have no evaluation test (like the huggingface Tutorial) we set a percentage of converting one from the train dataset. Here is 10% (common value)


seed = 42 #in finetunning we seet a seed in order to able to reproduce the results. add any number here.  If you do not want this. delete this four lines
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

################## Paths Setup ################
# here we set the folder that we want all files to be included. the code will download many files all will exist in this directory. is absolute path based 
# Path to save the rsults
workspace_dir = Path("/workspace")  #the folder all files will exist
experiment_dir = workspace_dir / "experiment" / name_of_experiment  #the folder that the experiments executions will be stored
experiment_dir.mkdir(parents=True, exist_ok=True)
cache_dir = workspace_dir / "database" #do not change this foder name. Create it on the folder before running the code
temp_dir = workspace_dir / "tmp" #do not change this folder name . Create it on the folder before running the code

# Set environment variables to use the custom directories
os.environ["TMPDIR"] = str(temp_dir)
os.environ["TEMP"] = str(temp_dir)
os.environ["TMP"] = str(temp_dir)
os.environ["HF_HOME"] = str(cache_dir)  # Hugging Face cache directory
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)  # Transformers cache directory

# Use tempfile to set a custom base directory for temporary files
tempfile.tempdir = str(temp_dir)


##################  Dataset Preparation ################
# Load the dataset with the specified cache directory
# you can use any dataset you want. in this epxepriment we use a 224X224 image size dataset. if you choose other size you have to change the ViT pretraind model to accept the size you used. Read on the HuggingFace about ViT pre-trained models that supports the transformer lib we use.

dataset = load_dataset("emre570/breastcancer-ultrasound-images", cache_dir=cache_dir)

train_val_split = dataset["train"].train_test_split(test_size=validation_dataset_percentage)

# Merge train and test sets
from datasets import DatasetDict

dataset = DatasetDict(
    {"train": train_val_split["train"], "validation": train_val_split["test"], "test": dataset["test"]}
)

# my datasets, Train, Validate, Test
train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]



# 1. Label Mapping: We convert between label IDs and their corresponding names, useful for model training and evaluation.
id2label = {id: label for id, label in enumerate(train_ds.features["label"].names)}
label2id = {label: id for id, label in id2label.items()}

# 2. Image Processing: Then, we utilize the ViTImageProcessor to standardize input image sizes and applies normalization specific to the pretrained model. Also, will define different transformations for training, validation, and testing to improve model generalization using torchvision.

model_name = "google/vit-large-patch16-224" # this is the model for this fine-tunning. Use eny model you want. Have to be in the same image sze as yout dataset (here 224)
processor = ViTImageProcessor.from_pretrained(model_name)
 
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

# Appling auto transformation rotation and crop to the dataset in order to have ore accurate fine-tunning

normalize = Normalize(mean=image_mean, std=image_std)

train_transforms = Compose([
    RandomResizedCrop(size),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])

val_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])

test_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])

# Create transform functions
def apply_train_transforms(examples):
    examples["pixel_values"] = [train_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples

def apply_val_transforms(examples):
    examples["pixel_values"] = [val_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples

def apply_test_transforms(examples):
    examples["pixel_values"] = [val_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples

# Apply transform functions to each set . all pixel values have been tranformed to tansors at that point
train_ds.set_transform(apply_train_transforms)
val_ds.set_transform(apply_val_transforms)
test_ds.set_transform(apply_test_transforms)


# Data Loading: Set up a custom collate function to properly batch images and labels, and create a DataLoader for efficient loading and batching during model training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)
val_dl = DataLoader(val_ds, collate_fn=collate_fn, batch_size=4)

# Batch Preparation: Retrieve and display the shape of data in a sample batch to verify correct processing and readiness for model input.
batch = next(iter(train_dl))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

##################  Training settings ################
#here you have to change the hyperparameter you optimize in the function input to the one you want each time. here is the learning_rate

def train_model(learning_rate):
    model = ViTForImageClassification.from_pretrained(
        model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    model.to(device)

    # Create specific directories for this combination of lr_scheduler_type and weight_decay
    var1 = f"lr_{learning_rate}" #a folder with this name will be created inside the experiment folder for each value of the experiment you run
    
    #these are folder names inside that folder
    output_dir = experiment_dir / var1  / "output-models"  #foder for the safetensor files and resluts
    log_dir = experiment_dir / var1  / "logs" # logs folder
    emission_dir = experiment_dir / var1  #emision folder
    
    # Setup the emissions tracker with the experiment name
    tracker = EmissionsTracker(output_dir=emission_dir, output_file=f"emissions_{var1}.csv")

    # Capture the start time for training
    start_time = time.time()

    tracker.start()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

#you can read more about this setting on the trainer on huggingface for the transformer lib
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=do_train,
        do_eval=do_eval,
        do_predict=do_predict,
        save_total_limit=2,
        report_to='tensorboard',
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=learning_rate,  #here you see we have no real valu but the variable name. this till take all the values we set in the begining. do that fot any hyperparameter you train. each time you must have only one variable and all rest have a real value.
        lr_scheduler_type='linear',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        load_best_model_at_end=True,
        logging_dir=str(log_dir),
        remove_unused_columns=False,
        logging_strategy="epoch",  
        logging_steps=10,  
        seed=seed,
        num_train_epochs=40,
    )


    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save the final model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Capture the end time
    end_time = time.time()
    emissions = tracker.stop()
    print(f"Emissions  for senario {name_of_experiment} : {emissions:.5f} kg CO2eq")


##################  Code Execution  ################
# Run training for each combination of learning rate scheduler type and weight decay sequentially
for lr in learning_rates:
    train_model(lr)
        

