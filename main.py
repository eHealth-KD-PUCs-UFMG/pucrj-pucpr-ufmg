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
EPOCH=50
BATCH_SIZE=1
PRETRAINED_MODEL = 'multilingual'
EARLY_STOP = 15
LEARNING_RATE=2e-5
trainset_name='training_develop'

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if PRETRAINED_MODEL == 'beto':
        trainset = json.load(open('data/original/ref/'+trainset_name+'/input_beto.json'))
        devset = json.load(open('data/original/ref/develop/input_beto.json'))
        model = Vicomtech(pretrained_model_path='dccuchile/bert-base-spanish-wwm-cased')
    elif PRETRAINED_MODEL == 'mt5':
        trainset = json.load(open('data/original/ref/'+trainset_name+'/input_mt5.json'))
        devset = json.load(open('data/original/ref/develop/input_mt5.json'))
        model = Vicomtech(pretrained_model_path='/scratch/thiago.ferreira/mt5')
    else:
        trainset = json.load(open('data/original/ref/'+trainset_name+'/input_multilingual.json'))
        devset = json.load(open('data/original/ref/develop/input_multilingual.json'))
        model = Vicomtech(pretrained_model_path='bert-base-multilingual-cased')
    model.to(device)
    
    criterion = nn.NLLLoss()

    initial_lr = LEARNING_RATE / 10
    lmbda = lambda epoch: min(10, epoch + 1)
    write_path = 'entity_model.pt'
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
    trainer = Train(model, criterion, optimizer, scheduler, trainset, devset, EPOCH, BATCH_SIZE, early_stop=EARLY_STOP, write_path=write_path, pretrained_model=PRETRAINED_MODEL, batch_status=BATCH_STATUS, task='entity')
    trainer.train()
    
    model = torch.load(write_path)
    initial_lr = LEARNING_RATE / 10
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
    lmbda = lambda epoch: min(10, epoch + 1)
    scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
    write_path = 'relation_model.pt'
    trainer = Train(model, criterion, optimizer, scheduler, trainset, devset, EPOCH, BATCH_SIZE, early_stop=EARLY_STOP, write_path=write_path, pretrained_model=PRETRAINED_MODEL, batch_status=BATCH_STATUS, task='relation')
    trainer.train()

    loss_func = MultiTaskLossWrapper(2)
    loss_func.to(device)
    loss_optimizer = optim.AdamW(loss_func.parameters(), lr=LEARNING_RATE)
    
#     write_path = '/scratch/thiago.ferreira/vicom_final/model.pt'
    EPOCH=100
    model = torch.load(write_path)
    initial_lr = LEARNING_RATE / 10
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
    lmbda = lambda epoch: min(10, epoch + 1)
    scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
    write_path = 'model.pt'
    trainer = Train(model, criterion, optimizer, scheduler, trainset, devset, EPOCH, BATCH_SIZE, early_stop=EARLY_STOP, write_path=write_path, pretrained_model=PRETRAINED_MODEL, batch_status=BATCH_STATUS, task='entity+relation', loss_func=loss_func, loss_optimizer=loss_optimizer)
    trainer.train()
    
    ## EVALUATION
    loss_func = MultiTaskLossWrapper(2)
    loss_func.to(device)
    loss_optimizer = optim.AdamW(loss_func.parameters(), lr=LEARNING_RATE)
    
    model = torch.load('/scratch/thiago.ferreira/vicom_final/model.pt')
    initial_lr = LEARNING_RATE / 10
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
    lmbda = lambda epoch: min(10, epoch + 1)
    scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
    trainer = Train(model, criterion, optimizer, scheduler, trainset, devset, EPOCH, BATCH_SIZE, early_stop=EARLY_STOP, write_path='', pretrained_model=PRETRAINED_MODEL, batch_status=BATCH_STATUS, task='entity+relation', loss_func=loss_func, loss_optimizer=loss_optimizer)
    
    trainer.eval_mode = 'training'
    trainer.eval_class_report()
    trainer.eval()
    
    trainer.eval_mode = 'develop'
    trainer.eval_class_report()
    trainer.eval()
    
    trainer.eval_mode = 'testing'
    trainer.eval_class_report()
    trainer.eval()