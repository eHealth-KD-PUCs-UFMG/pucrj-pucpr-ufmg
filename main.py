import torch
import torch.nn as nn
from torch import optim
from model import Vicomtech, MultiTaskLossWrapper
from sklearn.metrics import classification_report
import numpy as np
import json
import utils
from train import Train
import output
import postprocessing
from pathlib import Path
import time
from shutil import copyfile
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR

BATCH_STATUS=64
EPOCH=30
BATCH_SIZE=1
PRETRAINED_MODEL = 'multilingual'
EARLY_STOP = 10
LEARNING_RATE=2e-5

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if PRETRAINED_MODEL == 'beto':
        trainset = json.load(open('data/original/ref/training/input_beto.json'))
        devset = json.load(open('data/original/ref/develop/input_beto.json'))
        model = Vicomtech(pretrained_model_path='dccuchile/bert-base-spanish-wwm-cased')
    else:
        trainset = json.load(open('data/original/ref/training/input_multilingual.json'))
        devset = json.load(open('data/original/ref/develop/input_multilingual.json'))
        model = Vicomtech(pretrained_model_path='bert-base-multilingual-cased')
    model.to(device)

    initial_lr = LEARNING_RATE / 10
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
    lmbda = lambda epoch: min(10, epoch + 1)
    scheduler = LambdaLR(optimizer, lr_lambda=lmbda)

    loss_func = MultiTaskLossWrapper(2)
    loss_func.to(device)
    loss_optimizer = optim.AdamW(loss_func.parameters(), lr=LEARNING_RATE)
    
    criterion = nn.NLLLoss()

    trainer = Train(model, criterion, optimizer, scheduler, trainset, devset, EPOCH, BATCH_SIZE, early_stop=EARLY_STOP, pretrained_model=PRETRAINED_MODEL, batch_status=BATCH_STATUS, task='entity')
    trainer.train()
    
    model = torch.load('model.pt')
    initial_lr = LEARNING_RATE / 10
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
    lmbda = lambda epoch: min(10, epoch + 1)
    scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
    trainer = Train(model, criterion, optimizer, scheduler, trainset, devset, EPOCH, BATCH_SIZE, early_stop=EARLY_STOP, pretrained_model=PRETRAINED_MODEL, batch_status=BATCH_STATUS, task='entity+relation', loss_func=loss_func, loss_optimizer=loss_optimizer)
    trainer.train()