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
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
import torchvision


class Tabor(BackdoorDefense):
    name: str = 'tabor'
    
    #class Snooper:
    """
    A poison snooper for neural networks implementing the TABOR method.
    Named for: https://dune.fandom.com/wiki/Poison_snooper
    Based off of: https://github.com/bolunwang/backdoor/blob/master/visualizer.py
    """

    # upsample size, default is 1
    UPSAMPLE_SIZE = 1

    def __init__(self, args, model, batch_size=32, upsample_size=UPSAMPLE_SIZE, device='cpu') -> None:
        self.args = args
        self.device = device
        self.batch_size = batch_size
        self.model = model.to(device)
        self.loader = generate_dataloader(dataset=self.dataset, dataset_path=config.data_dir, batch_size=batch_size, split='val')
        self.return_logs = False
        self.steps = 200


        mask_size = np.ceil(np.array((self.input_size, self.input_size), dtype=float) /
                            upsample_size)
        mask_size = mask_size.astype(int)
        self.mask_size = mask_size
        self.setup_mask_and_pattern()


        self.hyperparameters = torch.tensor(np.array([1e-6, 1e-5, 1e-7, 1e-8, 0, 1e-2]), dtype=torch.float32).reshape(6, 1)
        self.opt = torch.optim.Adam([self.pattern_tanh_tensor, self.mask_tanh_tensor], lr=1e-3, betas=(0.5, 0.9))
    
    def setup_mask_and_pattern(self):
        mask = np.zeros(self.mask_size)
        pattern = np.zeros((self.input_size, self.input_size, 3))
        mask = np.expand_dims(mask, axis=2)

        mask_tanh = np.zeros_like(mask)
        pattern_tanh = np.zeros_like(pattern)

        # prepare mask related tensors
        self.mask_tanh_tensor = torch.tensor(mask_tanh, requires_grad=True)
        mask_tensor_unrepeat = (torch.tanh(self.mask_tanh_tensor) /
                                (2 - torch.finfo(torch.float32).eps) + 0.5)
        mask_tensor_unexpand = mask_tensor_unrepeat.repeat(1, 1, 3)
        self.mask_tensor = mask_tensor_unexpand.unsqueeze(0)
        upsample_layer = nn.Upsample(size=(upsample_size, upsample_size))
        mask_upsample_tensor_uncrop = upsample_layer(self.mask_tensor)
        uncrop_shape = mask_upsample_tensor_uncrop.shape[2:]

        self.upsample_layer = upsample_layer
        self.cropping_layer = nn.ZeroPad2d((0, uncrop_shape[1] - self.input_size, 0, uncrop_shape[0] - self.input_size))

        self.mask_upsample_tensor = cropping_layer(mask_upsample_tensor_uncrop)
        
        # prepare pattern related tensors
        self.pattern_tanh_tensor = torch.tensor(pattern_tanh, requires_grad=True)
        self.pattern_raw_tensor = (
            (torch.tanh(self.pattern_tanh_tensor) / (2 - torch.finfo(torch.float32).eps) + 0.5) *
            255.0)
    
    def upsample_mask(self, mask_tensor):
        mask_tensor_unrepeat = (torch.tanh(mask_tensor) /
                                (2 - torch.finfo(torch.float32).eps) + 0.5)
        mask_tensor_unexpand = mask_tensor_unrepeat.repeat(1, 1, 3)
        mask_tensor = mask_tensor_unexpand.unsqueeze(0)
        mask_upsample_tensor_uncrop = self.upsample_layer(mask_tensor)
        uncrop_shape = mask_upsample_tensor_uncrop.shape[2:]

        mask_upsample_tensor = self.cropping_layer(mask_upsample_tensor_uncrop)
        return mask_upsample_tensor
        

    def run(self):
        pattern_list = []
        mask_list = []
        loss_list = []
        for label in range(self.num_classes):
            mask_init = np.zeros(self.mask_size)
            pattern_init = np.zeros((self.input_size, self.input_size, 3))
            pattern_best, mask_best, mask_upsample_best, loss = self.snoop(label, pattern_init, mask_init) #TODO: Implement this function

            mask_list.append(mask_best)
            pattern_list.append(pattern_best)
            loss_list.append(loss)

            # np.savez(file_path, mark_list=[to_numpy(mark) for mark in mark_list],
            #          mask_list=[to_numpy(mask) for mask in mask_list],
            #          loss_list=loss_list)
            # print('Defense results saved at:', file_path)
            
            mark_path = os.path.normpath(os.path.join(
                self.folder_path, 'mark_tabor_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
            mask_path = os.path.normpath(os.path.join(
                self.folder_path, 'mask_tabor_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
            trigger_path = os.path.normpath(os.path.join(
                self.folder_path, 'trigger_tabor_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
            save_image(pattern_best, mark_path)
            save_image(mask_best, mask_path)
            save_image(mask_best * pattern_best, trigger_path)
            
            print('Restored trigger mark of class %d saved at:' % label, mark_path)
            print('Restored trigger mask of class %d saved at:' % label, mask_path)
            print('Restored trigger of class %d saved at:' % label, trigger_path)
            print('')
        return pattern_list, mask_list, loss_list

    def train_step(self, input_tensor, y_true_tensor, y_target_tensor):
        self.opt.zero_grad()
        self.mask_upsample_tensor = self.upsample_mask(self.mask_tensor)

        input_raw_tensor = input_tensor
        reverse_mask_tensor = (torch.ones_like(self.mask_upsample_tensor) -
                               self.mask_upsample_tensor)

        # IMPORTANT: MASK OPERATION IN RAW DOMAIN -> TENSORFLOW WAY OF DOING IT? TODO: MOVE TO FUNCTION?
        X_adv_raw_tensor = (
            reverse_mask_tensor * input_raw_tensor +
            self.mask_upsample_tensor * self.pattern_raw_tensor)

        X_adv_tensor = X_adv_raw_tensor

        output_tensor = self.model(X_adv_tensor)
        self.loss_ce = nn.functional.cross_entropy(output_tensor, y_target_tensor.argmax(dim=1))
        self.loss_reg = self.build_tabor_regularization(input_raw_tensor,
                                                self.model, y_target_tensor,
                                                y_true_tensor)
        self.loss_reg = torch.matmul(self.loss_reg.view(1, 6), self.hyperparameters)
        self.loss = torch.mean(self.loss_ce) + self.loss_reg

        self.loss.backward()
        self.opt.step()
        return self.loss_ce, self.loss_reg, self.loss

    def build_tabor_regularization(self, input_raw_tensor, model,
                                   y_target_tensor, y_true_tensor):
        reg_losses = []

        # R1 - Overly large triggers
        mask_l1_norm = torch.sum(torch.abs(self.mask_upsample_tensor))
        mask_l2_norm = torch.sum(torch.square(self.mask_upsample_tensor))
        mask_r1 = (mask_l1_norm + mask_l2_norm)

        pattern_tensor = (torch.ones_like(self.mask_upsample_tensor) -
                          self.mask_upsample_tensor) * self.pattern_raw_tensor
        pattern_l1_norm = torch.sum(torch.abs(pattern_tensor))
        pattern_l2_norm = torch.sum(torch.square(pattern_tensor))
        pattern_r1 = (pattern_l1_norm + pattern_l2_norm)

        # R2 - Scattered triggers
        pixel_dif_mask_col = torch.sum(torch.square(
            self.mask_upsample_tensor[:, :-1, :] -
            self.mask_upsample_tensor[:, 1:, :]))
        pixel_dif_mask_row = torch.sum(torch.square(
            self.mask_upsample_tensor[:-1, :, :] -
            self.mask_upsample_tensor[1:, :, :]))
        mask_r2 = pixel_dif_mask_col + pixel_dif_mask_row

        pixel_dif_pat_col = torch.sum(torch.square(pattern_tensor[:, :-1, :] -
                                                   pattern_tensor[:, 1:, :]))
        pixel_dif_pat_row = torch.sum(torch.square(pattern_tensor[:-1, :, :] -
                                                   pattern_tensor[1:, :, :]))
        pattern_r2 = pixel_dif_pat_col + pixel_dif_pat_row

        # R3 - Blocking triggers
        cropped_input_tensor = (torch.ones_like(self.mask_upsample_tensor) -
                                self.mask_upsample_tensor) * input_raw_tensor
        r3 = torch.mean(nn.functional.cross_entropy(model(cropped_input_tensor), y_true_tensor[0].view(1, -1)))

        # R4 - Overlaying triggers
        mask_crop_tensor = self.mask_upsample_tensor * self.pattern_raw_tensor
        r4 = torch.mean(nn.functional.cross_entropy(model(mask_crop_tensor), y_target_tensor[0].view(1, -1)))

        reg_losses.append(mask_r1)
        reg_losses.append(pattern_r1)
        reg_losses.append(mask_r2)
        reg_losses.append(pattern_r2)
        reg_losses.append(r3)
        reg_losses.append(r4)

        return torch.stack(reg_losses)

    def reset_opt(self):
        self.opt.zero_grad()

    def reset_state(self, pattern_init, mask_init):
        print('resetting state')

        # setting mask and pattern
        mask = np.array(mask_init)
        pattern = np.array(pattern_init)
        mask = np.clip(mask, 0, 1)
        pattern = np.clip(pattern, 0, 255)
        mask = np.expand_dims(mask, axis=2)

        # convert to tanh space
        mask_tanh = np.arctanh((mask - 0.5) * (2 - np.finfo(np.float32).eps))
        pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - np.finfo(np.float32).eps))
        print('mask_tanh', np.min(mask_tanh), np.max(mask_tanh))
        print('pattern_tanh', np.min(pattern_tanh), np.max(pattern_tanh))

        self.mask_tanh_tensor.data = torch.tensor(mask_tanh, requires_grad=True)
        self.pattern_tanh_tensor.data = torch.tensor(pattern_tanh, requires_grad=True)

        # resetting optimizer states
        self.reset_opt()

    def snoop(self, y_target, pattern_init, mask_init):
        self.reset_state(pattern_init, mask_init)

        # best optimization results
        mask_best = None
        mask_upsample_best = None
        pattern_best = None
        Y_target = None
        loss_best = float('inf')

        # logs and counters for adjusting balance cost
        logs = []

        # loop start
        for step in range(self.steps):

            # record loss for all mini-batches
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            for data in iter(self.loader):#range(ceil(len(x) / self.batch_size)):
                X_batch = data[0].to(self.device) #x[idx * self.batch_size:(idx + 1) * self.batch_size]
                Y_batch = data[1].to(self.device) #y[idx * self.batch_size:(idx + 1) * self.batch_size]
                if Y_target is None:
                    Y_target = torch.eye(self.num_classes)[y_target].repeat(Y_batch.shape[0], 1).to(self.device)

                (loss_ce_value,
                 loss_reg_value,
                 loss_value) = self.train_step(X_batch, Y_batch, Y_target)
                loss_ce_list.extend(loss_ce_value.flatten().detach().numpy())
                loss_reg_list.extend(loss_reg_value.flatten().detach().numpy())
                loss_list.extend(loss_value.flatten().detach().numpy())

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)

            # check to save best mask or not
            if avg_loss < loss_best:
                mask_best = self.mask_tensor.detach().numpy()[0, ..., 0]
                mask_upsample_best = self.mask_upsample_tensor.detach().numpy()[0, ..., 0]
                pattern_best = self.pattern_raw_tensor.detach().numpy()
                loss_best = avg_loss
                # with open('pattern.npy', 'wb') as f:
                #     np.save(f, pattern_best)
                # with open('mask.npy', 'wb') as f:
                #     np.save(f, mask_best)

            # save log
            logs.append((step, avg_loss_ce, avg_loss_reg, avg_loss))
            print("Step {} | loss_ce {} | loss_reg {} | loss {}".format(step, avg_loss_ce, avg_loss_reg, avg_loss))

        # save the final version
        if mask_best is None:
            mask_best = self.mask_tensor.detach().numpy()[0, ..., 0]
            mask_upsample_best = self.mask_upsample_tensor.detach().numpy()[0, ..., 0]
            pattern_best = self.pattern_raw_tensor.detach().numpy()

        if self.return_logs:
            return pattern_best, mask_best, mask_upsample_best, lost_best, logs
        else:
            return pattern_best, mask_best, mask_upsample_best, loss_best
    
    def detect(self):
        mark_list, mask_list, loss_list = self.run()
        mask_norms = mask_list.flatten(start_dim=1).norm(p=1, dim=1)
        print('mask norms: ', mask_norms)
        print('mask anomaly indices: ', normalize_mad(mask_norms))
        print('loss: ', loss_list)
        print('loss anomaly indices: ', normalize_mad(loss_list))

        anomaly_indices = normalize_mad(mask_norms)
        # overlap = jaccard_idx(mask_list[self.target_class], self.trigger_mask,
        #                         select_num=(self.trigger_mask > 0).int().sum())
        # print(f'Jaccard index: {overlap:.3f}')
        
        # self.suspect_class = torch.argmin(mask_norms).item()
        suspect_classes = []
        suspect_classes_anomaly_indices = []
        if self.oracle:
            print("<Oracle> Unlearning with reversed trigger from class %d" % self.suspect_class)
            self.unlearn()
        else:
            for i in range(self.num_classes):
                if mask_norms[i] > torch.median(mask_norms): continue
                if anomaly_indices[i] > 2:
                    suspect_classes.append(i)
                    suspect_classes_anomaly_indices.append(anomaly_indices[i])
            print("Suspect Classes:", suspect_classes)
            if len(suspect_classes) > 0:
                max_idx = torch.tensor(suspect_classes_anomaly_indices).argmax().item()
                self.suspect_class = suspect_classes[max_idx]
                print("Unlearning with reversed trigger from class %d" % self.suspect_class)
                self.unlearn()

    def unlearn(self):
        # label = config.target_class[self.args.dataset]
        label = self.suspect_class
        mark_path = os.path.normpath(os.path.join(
            self.folder_path, 'mark_neural_cleanse_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
        mask_path = os.path.normpath(os.path.join(
            self.folder_path, 'mask_neural_cleanse_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
        trigger_path = os.path.normpath(os.path.join(
            self.folder_path, 'trigger_neural_cleanse_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
        
        mark = Image.open(mark_path).convert("RGB")
        mark = transforms.ToTensor()(mark)
        mask = Image.open(mask_path).convert("RGB")
        mask = transforms.ToTensor()(mask)[0]
        print(mark.shape, mask.shape)

        if self.args.dataset == 'cifar10':
            clean_set_dir = os.path.join('clean_set', self.args.dataset, 'clean_split')
            clean_set_img_dir = os.path.join(clean_set_dir, 'data')
            clean_set_label_path = os.path.join(clean_set_dir, 'clean_labels')
            full_train_set = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                                        label_path=clean_set_label_path, transforms=transforms.ToTensor())
            # full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'), train=True, download=True, transform=transforms.ToTensor())
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            batch_size = 128
            lr = 0.01
            if 'resnet110' in supervisor.get_arch(self.args).__name__:
                # for SRA attack
                lr = 0.001
        elif self.args.dataset == 'gtsrb':
            clean_set_dir = os.path.join('clean_set', self.args.dataset, 'clean_split')
            clean_set_img_dir = os.path.join(clean_set_dir, 'data')
            clean_set_label_path = os.path.join(clean_set_dir, 'clean_labels')
            full_train_set = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                                        label_path=clean_set_label_path, transforms=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
            # full_train_set = datasets.GTSRB(os.path.join(config.data_dir, 'gtsrb'), split='train', download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
            data_transform_aug = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            batch_size = 128
            lr = 0.002
            
            if self.args.poison_type == 'BadEncoder':
                data_transform_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(32, 4),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
                lr = 0.0001
        elif self.args.dataset == 'imagenet':
            from utils import imagenet
            # train_set_dir = os.path.join(config.imagenet_dir, 'train')
            clean_set_dir = os.path.join(config.imagenet_dir, 'val')
            full_train_set = imagenet.imagenet_dataset(directory=clean_set_dir, data_transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256)), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]),
                                                       poison_directory=None, poison_indices=None, target_class=config.target_class['imagenet'], num_classes=1000)
            
            clean_split_meta_dir = os.path.join('clean_set', self.args.dataset, 'clean_split')
            clean_indices = torch.load(os.path.join(clean_split_meta_dir, 'clean_split_indices'))
            full_train_set = torch.utils.data.Subset(full_train_set, clean_indices)
            
            data_transform_aug = transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            batch_size = 256
            lr = 0.01 # IMAGENET1K_V1
            # lr = 0.001 # ViT, IMAGENET1K_SWAG_LINEAR_V1
        else:
            raise NotImplementedError()
        train_data = DatasetCL(1.0, full_dataset=full_train_set, transform=data_transform_aug, poison_ratio=0.2, mark=mark, mask=mask)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.module.parameters(), lr, momentum=self.momentum, weight_decay=self.weight_decay)

        val_atk(self.args, self.model)
        
        for epoch in range(1):  # train backdoored base model
            # Train
            self.model.train()
            preds = []
            labels = []
            for data, target in tqdm(train_loader):
                optimizer.zero_grad()
                data, target = data.cuda(), target.cuda()  # train set batch
                output = self.model(data)
                preds.append(output.argmax(dim=1))
                labels.append(target)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            train_acc = (torch.eq(preds, labels).int().sum()) / preds.shape[0]
            print('\n<Unlearning> Train Epoch: {} \tLoss: {:.6f}, Train Acc: {:.6f}, lr: {:.2f}'.format(epoch, loss.item(), train_acc, optimizer.param_groups[0]['lr']))
            val_atk(self.args, self.model)
            
        torch.save(self.model.module.state_dict(), supervisor.get_model_dir(self.args, defense=True))
        print("Saved repaired model to {}".format(supervisor.get_model_dir(self.args, defense=True)))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--checkpoint', type=str)
#     args = parser.parse_args()

#     model = build_model()
#     model.load_weights(args.checkpoint)

#     pattern = np.random.random((self.input_size, self.input_size, 3)) * 255.0
#     mask = np.random.random((self.input_size, self.input_size))
#     dataset = GTSRBDataset()

#     x = np.concatenate([dataset.train_images, dataset.test_images])
#     y = np.concatenate([dataset.train_labels, dataset.test_labels])

#     snooper = Snooper(model)
#     pattern_best, mask_best, mask_upsample_best = snooper.snoop(x, y, 33, pattern, mask)
    
#     with open('pattern.npy', 'wb') as f:
#         np.save(f, pattern_best)
#     with open('mask.npy', 'wb') as f:
#         np.save(f, mask_best)

#     for x in [pattern_best, mask_upsample_best, mask_best]:
#         print(x.shape)