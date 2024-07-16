import sys, os
from tkinter import E

import torch.utils
import torch.utils.data
EXT_DIR = ['..']
for DIR in EXT_DIR:
    if DIR not in sys.path: sys.path.append(DIR)

import numpy as np
import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import config
import torch.optim as optim
import time
import datetime
from tqdm import tqdm
from .tools import AverageMeter, generate_dataloader, tanh_func, to_numpy, jaccard_idx, normalize_mad, val_atk
from . import BackdoorDefense
from utils import supervisor, tools, resnet
import random
from copy import deepcopy

class NeuralCleanseGeneralized(BackdoorDefense):
    name: str = 'NCG'

    def __init__(self, args, epoch=50, batch_size=32, treshold=2, device='cpu', mitigation=True) -> None:
        super().__init__(args)

        self.args = args

        self.classes = range(self.num_classes)
        self.logger = PrintLogger()
        self.treshold = treshold
        self.device = device
        self.input_size = config.get_feature_size(self.model)
        self.results = {}
        self.epochs = epoch
        self.acts = {}
        self.mitigation = mitigation
        self.target_label = None

        self.attack_succ_threshold = 0.99
        self.init_cost = 0.01
        self.cost_multiplier_up = 1.5
        self.cost_multiplier_down = 1.1
        self.patience = 5

        self.folder_path = 'other_defenses_tool_box/results/NCG'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)

        self.loader = generate_dataloader(dataset=self.dataset, dataset_path=config.data_dir, batch_size=batch_size, split='val')
        self.tqdm = False
        self.suspect_class = config.target_class[args.dataset] # default with oracle
        self.layer = config.get_layer(self.model)
        self.train_args = {"batch_size": batch_size, 'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        self.criterion = nn.CrossEntropyLoss()

    
    def detect(self):
        features = activations_from_data(self.model, self.loader, self.layer, 'features', self.device, self.get_activation)
        self.features = torch.utils.data.DataLoader(ActivationsDataset(features), **self.train_args)
        self.run(make_submodel(self.model), self.features)
        self.logger.log("Neural Cleanse General finished")
        for target in self.suspect_labels:
            self.logger.log(f"Target: {target}, Trigger norm: {self.results[target][1]}")
            trigger, mask = self.results[target][0]
            trigger = torch.tensor(trigger, device=self.device, requires_grad=False)
            mask = torch.tensor(mask, device=self.device, requires_grad=False)
            retrained_model = self.retrain(make_submodel(self.model), self.features, trigger, mask, self.device, self.criterion, epochs=30, lr=0.01, amp=1)
            self.retrained_model = retrained_model
            break
        
        self.model = update_model(self.model, self.retrained_model) if hasattr(self, 'retrained_model') else self.model
        torch.save(self.model.module.state_dict(), supervisor.get_model_dir(self.args, defense=True))
        self.logger.log("Saved repaired model to {}".format(supervisor.get_model_dir(self.args, defense=True)))
        
        

    def run(self, model, clean_dataset):
        for target in self.classes:
            trigger, mask, trigger_norm, avg = self.optimize_minimal_trigger(model, clean_dataset, target)
            self.logger.log(f"Target: {target}, Trigger norm: {trigger_norm}, Accuracy: {avg}")
            if avg >= 0.95:
                self.results[target] = ((trigger, mask), trigger_norm)
        
        mean_norm = np.mean([result[1] for result in self.results.values()])
        self.results = self.detect_outliers(self.results, mean_norm)
        suspect_labels = [target for target, result in self.results.items() if result[2] > self.treshold and result[1] < mean_norm]
        suspect_labels = sorted(suspect_labels, key=lambda x: self.results[x][2], reverse=True)
        self.suspect_labels = suspect_labels

        # self.logger.log(f"Neural Cleanse results: {self.results}")
        formated_suspect_labels = ', '.join([f"{t}: {self.results[t][2]}" for t in suspect_labels])
        self.logger.log(f"Suspect labels: {formated_suspect_labels}")

        suspect_labels_high = [target for target, result in self.results.items() if result[2] > self.treshold*2 and result[1] > mean_norm]
        suspect_labels_high = sorted(suspect_labels_high, key=lambda x: self.results[x][2], reverse=True)
        if len(suspect_labels_high) > 0:
            formated_suspect_labels_high = ', '.join([f"{t}: {self.results[t][2]}" for t in suspect_labels_high])
            self.logger.log(f"Suspect labels high: {formated_suspect_labels_high}")
            self.suspect_labels = suspect_labels_high
        
        if not self.mitigation:
            return

        #NEW
        if len(self.suspect_labels) > 0:
            self.results_round2 = {}
            self.epochs *= 2
            for target in self.suspect_labels:
                trigger, mask, trigger_norm, acc = self.optimize_minimal_trigger(model, clean_dataset, target)
                self.logger.log(f"Target: {target}, Trigger norm: {trigger_norm}, Accuracy: {acc}")
                self.results_round2[target] = ((trigger, mask), trigger_norm)

            suspect_labels2 = sorted(self.results_round2.keys(), key=lambda x: self.results_round2[x][1], reverse=False)
            self.target_label = suspect_labels2[0]
            self.logger.log(f"Target label round 2: {self.target_label}")
        else:
            self.target_label = None

    def optimize_minimal_trigger(self, model, clean_dataloader, target):
        mask = torch.zeros(self.input_size)#, requires_grad=True)
        trigger = torch.zeros(self.input_size)#, requires_grad=True)
        
        mask = mask.unsqueeze(0).to(self.device)
        trigger = trigger.unsqueeze(0).to(self.device)

        mask = mask.requires_grad_(True)
        trigger = trigger.requires_grad_(True)
        model.eval()
        model = model.requires_grad_(False)
        #optimizer = torch.optim.AdamW([trigger, mask], lr=0.1, weight_decay=0.05)
        optimizer = torch.optim.Adam([trigger, mask], lr=0.1, betas=(0.9, 0.9)) #, lr=0.5)
        acc = AverageMeter("Accuracy")

        loss_fn = torch.nn.CrossEntropyLoss()
        l = 0.1
        l2 = 0.1

        cost, cost_up_counter, cost_down_counter, cost_up_flag, cost_down_flag, cost_set_counter = self.init_cost, 0, 0, False, False, 0
        epochs = self.epochs
        should_break = False
        norm = torch.nn.LayerNorm(self.input_size, device=self.device)
        r = (pbar := tqdm(range(epochs), total=epochs, ncols=100)) if self.tqdm else range(epochs)
        for _ in r:
            correct = 0
            acc.reset()
            for _, (data, y) in enumerate(clean_dataloader):
                data = data.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()

                # m = torch.clip(mask, 0, 1)
                # t = torch.clip(trigger, 0, 1)
                m = tanh_func(mask) # torch.sigmoid(mask) # torch.tanh(mask) / 2 + 0.5
                t = trigger #tanh_func(trigger) #torch.sigmoid(trigger) # trigger
                # t = torch.sigmoid(trigger)
                #print(data.shape)
                x_hat = data * (1 - m) + t * m
                x_hat = norm(x_hat)
                y_hat = model(x_hat)
                yt = torch.ones_like(y) * target

                loss = loss_fn(y_hat, yt) + cost * torch.norm(m, p=1) + cost * torch.norm(t, p=2)
                loss.backward()
                optimizer.step()
                
                pred = y_hat.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                c = pred.eq(yt.view_as(pred)).float().mean().item()
                acc.update(c)
                correct += pred.eq(yt.view_as(pred)).sum().item()
            c = correct / len(clean_dataloader.dataset)
            if self.tqdm:
                pbar.set_description(f'Loss: {loss.item():.4f} \tCorrect: {c:.4f}')
            cost, cost_up_counter, cost_down_counter, cost_up_flag, cost_down_flag, cost_set_counter = self.modify_cost(acc, cost, cost_up_counter, cost_down_counter, cost_up_flag, cost_down_flag, cost_set_counter)
        m = tanh_func(mask) #torch.sigmoid(mask) #torch.tanh(mask) / 2 + 0.5
        t = trigger #tanh_func(trigger) #torch.sigmoid(trigger) #torch.tanh(trigger) / 2 + 0.5
        return (t).detach().cpu().numpy()[0], m.detach().cpu().numpy()[0], torch.norm((t*m).detach(), p=1).cpu().numpy(), acc.avg

    
    def hook(self, model, input, output):
        self.features[self.name] = output.detach()

    def get_activation(self, name, features):
        self.features = features
        self.name = name
        return self.hook       

    def detect_outliers(self, results: dict, mean_norm):
        mad = np.median([np.abs(result[1] - mean_norm) for result in results.values()])
        
        for target, result in results.items():
            results[target] = (result[0], result[1], np.abs(result[1] - mean_norm) / mad)

        return results
    
    def retrain(self, model, train_dataloader, trigger, mask, device, criterion, epochs=5, lr=0.0005, amp=1):
        model.requires_grad_(True)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        for _ in tqdm(range(epochs)):
            for data in iter(train_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                if trigger is not None:
                    # m = torch.where(mask > 0.5, torch.ones_like(mask), torch.zeros_like(mask))
                    i = int(0.2*inputs.shape[0])
                    inputs[:i] = inputs[:i] * (1-mask) + amp * mask * trigger
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
        return model

    def modify_cost(self, acc, cost, cost_up_counter, cost_down_counter, cost_up_flag, cost_down_flag, cost_set_counter):
        # check cost modification
        if cost == 0 and acc.avg >= self.attack_succ_threshold:
            cost_set_counter += 1
            if cost_set_counter >= self.patience:
                cost = self.init_cost
                cost_up_counter = 0
                cost_down_counter = 0
                cost_up_flag = False
                cost_down_flag = False
                print('initialize cost to %.2f' % cost)
        else:
            cost_set_counter = 0

        if acc.avg >= self.attack_succ_threshold:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1

        if cost_up_counter >= self.patience:
            cost_up_counter = 0
            print('up cost from %.4f to %.4f' % (cost, cost * self.cost_multiplier_up))
            cost *= self.cost_multiplier_up
            cost_up_flag = True
        elif cost_down_counter >= self.patience:
            cost_down_counter = 0
            print('down cost from %.4f to %.4f' % (cost, cost / self.cost_multiplier_down))
            cost /= self.cost_multiplier_down
            cost_down_flag = True
        return cost, cost_up_counter, cost_down_counter, cost_up_flag, cost_down_flag, cost_set_counter

class ActivationsDataset(Dataset):
    def __init__(self, activations_classes):
        self.activations = []
        self.labels = []
        for k, v in activations_classes.items():
            for a in v[0]:
                self.activations.append(a)
                self.labels.append(k)

    def __getitem__(self, index):
        return torch.tensor(self.activations[index], requires_grad=False), self.labels[index]

    def __len__(self):
        return len(self.activations)
    
class PrintLogger:
    def log(self, msg):
        print(msg)

def activations_from_data(model, dataset, layer, name, device, get_activation):
    model = model.to(device)
    model.eval()

    activations = {}
    layer.register_forward_hook(get_activation(name, features=activations))

    a = {}
    with torch.no_grad():
        hs = []
        ys = []
        for _, (data, y) in tqdm(enumerate(dataset)):
            data = data.to(device)

            output = model(data)
            h = activations[name]
            h = h.view(h.size(0), -1)
            hs.append(h.cpu().numpy())
            ys.append(y.cpu().numpy())
    ss = np.concatenate(hs, axis=0)
    ys = np.concatenate(ys, axis=0)

    for si, y in zip(enumerate(ss), ys):
        if y not in a:
            a[y] = []
        a[y].append(si)
    for k,v in a.items():
        a[k] = (np.array([x[1] for x in v]), np.array([x[0] for x in v]))

    return a

class SubPreResNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.linear = model.linear
    def forward(self, x):
        return self.linear(x)

class SubResNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc = model.fc
    def forward(self, x):
        return self.fc(x)

def make_submodel(model) -> nn.Module:
    model = deepcopy(model)
    dp = None
    if type(model) == nn.DataParallel:
        dp = model
        model = dp.module
    
    if model.__class__.__name__.startswith('Pre'):
        model = SubPreResNet(model)
    elif type(model) == resnet.ResNet:
        model = SubPreResNet(model)
    else:
        model = SubResNet(model)

    if dp is not None:
        dp.module = model
        return dp
    return model

def update_model(model, new_model):
    model = deepcopy(model)
    dp = None
    if type(model) == nn.DataParallel:
        dp = model
        model = dp.module

    if type(new_model) == nn.DataParallel:
        new_model = new_model.module

    if new_model.__class__.__name__.startswith('SubPre'):
        model.linear = new_model.linear
    else:
        model.fc = new_model.fc
    
    if dp is not None:
        dp.module = model
        return dp
    return model