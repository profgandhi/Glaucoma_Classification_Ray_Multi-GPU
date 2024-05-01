from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from tqdm import tqdm

from steps.models import Model

from steps.dataset_loaders import get_dataloaders
import os
from typing import Dict
import ray.train.torch
from torch import nn
from torch import optim
import torch

from steps.custom_loss import FocalLoss

from sklearn.metrics import confusion_matrix

def train_func_per_worker(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
    name = "VGG16"

    # Get dataloaders inside the worker training function
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size)

    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

    name = "Custom_VIT"
    model = Model(name).get_model()

    model = ray.train.torch.prepare_model(model)

    #loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = FocalLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Model training loop
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        model.train()
        for X, y,features in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            pred = model(X.float(),features.float())
            loss = loss_fn(pred, y)
            ray.train.report(metrics={"train_step_loss": loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss, num_correct, num_total = 0, 0, 0
        predicted_labels = []
        true_labels = []
        with torch.no_grad():
            for X, y, features in tqdm(test_dataloader, desc=f"Test Epoch {epoch}"):
                pred = model(X.float(),features.float())
                loss = loss_fn(pred, y)
                
                test_loss += loss.item()
                num_total += y.shape[0]
                num_correct += (pred.argmax(1) == y).sum().item()
                
                #Confusion matrix
                predicted_labels.extend(pred.argmax(1).cpu().numpy())  # Append predicted class indices
                true_labels.extend(y.cpu().numpy())

        test_loss /= len(test_dataloader)
        accuracy = num_correct / num_total
        mat = confusion_matrix(true_labels, predicted_labels)
        print(mat)


        # [3] Report metrics to Ray Train
        # ===============================
        ray.train.report(metrics={"loss": test_loss, "accuracy": accuracy,'Confusion_matrix':mat})
        