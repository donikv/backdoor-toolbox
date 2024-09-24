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
from utils.unet_model import UNet


import torch
import os
from copy import deepcopy
import random
import pandas as pd
from tqdm import tqdm

class MaskGenerator(nn.Module):
    def __init__(self, init_mask, classifier) -> None:
        super().__init__()
        self._EPSILON = 1e-7
        self.classifier = classifier
        self.mask_tanh = nn.Parameter(init_mask.clone().detach().requires_grad_(True))
    
    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        bounded = mask / (2 + self._EPSILON) + 0.5
        return bounded

class BTIDBFU(BackdoorDefense):
    name: str = 'BTI-DBFU'

    def __init__(self, args, epoch=50, batch_size=32, device='cpu', pretrained_path='models', gen_lr=1e-3) -> None:
        super().__init__(args)

        self.args = args
        self.device = device

        self.classifier = self.model if not isinstance(self.model, nn.DataParallel) else self.model.module
        self.classifier = self.classifier.to(device)
        self.opt_cls = torch.optim.Adam(self.classifier.parameters(), lr=1e-4)

        self.cln_trainloader = generate_dataloader(dataset=self.dataset, dataset_path=config.data_dir, batch_size=batch_size, split='val')

        self.gen_lr = gen_lr
        bd_gen = UNet(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4)
        self.bd_gen = bd_gen.to(device)
        self.opt_bd = torch.optim.Adam(bd_gen.parameters(), lr=self.gen_lr)
        self.bd_gen.eval()

        self.nround = epoch
        
        self.mse = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()

        self.detected_tlabel = None

        self.tlabel = config.target_class[args.dataset] # default with oracle
        self.mround = 1
        self.uround = 1
        self.ul_round = 1
        self.norm_bound = 0.3
        self.feat_bound = 3
        self.earlystop = True 
        self.size = self.img_size

    def detect(self):
        classifier, bd_gen = self.classifier, self.bd_gen
        cln_trainloader = self.cln_trainloader

        for n in range(self.nround):
            self.reverse(classifier, bd_gen, n)
            if n == 0:
                self.detected_tlabel = self.get_target_label(testloader=cln_trainloader, testmodel=classifier, midmodel=bd_gen)
            elif self.earlystop:
                checked_tlabel = self.get_target_label(testloader=cln_trainloader, testmodel=classifier, midmodel=bd_gen)
                if checked_tlabel != self.detected_tlabel:
                    break
            self.classifier = self.unlearn(classifier, bd_gen, n)
        
        # self.model = update_model(self.model, self.retrained_model) if hasattr(self, 'retrained_model') else self.model
        torch.save(self.classifier.state_dict(), supervisor.get_model_dir(self.args, defense=True))
        print("Saved repaired model to {}".format(supervisor.get_model_dir(self.args, defense=True)))

    def test(self, testloader, testmodel, box, poisoned=False, poitarget=False , midmodel = None, passlabel=None, feat_mask=None, name="BA"):
        model = deepcopy(testmodel)
        model.eval()        
        correct = 0
        total = 0

        if poisoned:
            param1, param2, _ = box.get_state_dict()

        pbar = tqdm(testloader, desc="Test")

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(box.device), targets.to(box.device)
                ori_target = targets
                if poisoned:
                    inputs = box.poisoned(inputs, param1, param2)

                if not midmodel is None:
                    tmp_model = deepcopy(midmodel)
                    tmp_model.eval()
                    gnoise = 0.03 * torch.randn_like(inputs, device=box.device)
                    inputs = tmp_model(inputs + gnoise)
                    del tmp_model

                if poitarget:
                    if box.attack_type == "all2all":
                        targets = torch.remainder(targets+1, box.num_classes).to(box.device)
                    elif box.attack_type == "all2one":
                        targets = torch.ones_like(targets, device=box.device) * box.tlabel

                if not feat_mask is None:
                    feat = model.from_input_to_features(inputs)
                    outputs = model.from_features_to_output(feat_mask*feat)
                else:
                    outputs = model(inputs)

                _, predicted = outputs.max(1)

                for i in range(inputs.shape[0]):
                    if (not passlabel is None) and ori_target[i] == passlabel:
                        continue
                    total += 1
                    p = predicted[i]
                    t = targets[i]
                    if p == t:
                        correct += 1

                if total > 0:
                    acc = 100.*correct/total
                else:
                    acc = 0

                pbar.set_postfix({name: "{:.4f}".format(acc)})

        return 100.*correct/total

    def get_target_label(self, testloader, testmodel, midmodel = None):
        model = deepcopy(testmodel)
        model.eval()        
        reg = np.zeros([self.num_classes])
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if not midmodel is None:
                    tmodel = deepcopy(midmodel)
                    tmodel.eval()
                    gnoise = 0.03 * torch.randn_like(inputs, device=self.device)
                    inputs = tmodel(inputs + gnoise)

                outputs = model(inputs)
                _, predicted = outputs.max(1)

                for i in range(inputs.shape[0]):
                    p = predicted[i]
                    reg[p] += 1
                    # t = targets[i]
                    # if p == t:
                    #     reg[t] += 1
                        
        return np.argmax(reg)


    def reverse(opt, model, bd_gen, n):
        inv_classifier = deepcopy(model)
        inv_classifier.eval()
        tmp_img = torch.ones([1, 3, opt.size, opt.size], device=opt.device)
        tmp_feat = inv_classifier.from_input_to_features(tmp_img)
        feat_shape = tmp_feat.shape
        init_mask = torch.randn(feat_shape).to(opt.device)
        m_gen = MaskGenerator(init_mask=init_mask, classifier=inv_classifier)
        opt_m = torch.optim.Adam([m_gen.mask_tanh], lr=0.01)
        for m in range(opt.mround):
            tloss = 0
            tloss_pos_pred = 0
            tloss_neg_pred = 0
            m_gen.train()
            inv_classifier.train()
            pbar = tqdm(opt.cln_trainloader, desc="Decoupling Benign Features")
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                opt_m.zero_grad()
                cln_img = cln_img.to(opt.device)
                targets = targets.to(opt.device)
                feat_mask = m_gen.get_raw_mask()
                cln_feat = inv_classifier.from_input_to_features(cln_img)
                mask_pos_pred = inv_classifier.from_features_to_output(feat_mask*cln_feat)
                remask_neg_pred = inv_classifier.from_features_to_output((1-feat_mask)*cln_feat)
                mask_norm = torch.norm(feat_mask, 1)

                loss_pos_pred = opt.ce(mask_pos_pred, targets)
                loss_neg_pred = opt.ce(remask_neg_pred, targets)            
                loss = loss_pos_pred - loss_neg_pred

                loss.backward()
                opt_m.step()

                tloss += loss.item()
                tloss_pos_pred += loss_pos_pred.item()
                tloss_neg_pred += loss_neg_pred.item()
                pbar.set_postfix({"round": "{:d}".format(n), 
                                "epoch": "{:d}".format(m),
                                "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                                "loss_pos_pred": "{:.4f}".format(tloss_pos_pred/(batch_idx+1)),
                                "loss_neg_pred": "{:.4f}".format(tloss_neg_pred/(batch_idx+1)),
                                "mask_norm": "{:.4f}".format(mask_norm)})
                
        feat_mask = m_gen.get_raw_mask().detach()

        for u in range(opt.uround):
            tloss = 0
            tloss_benign_feat = 0
            tloss_backdoor_feat = 0
            tloss_norm = 0
            m_gen.eval()
            bd_gen.train()
            inv_classifier.eval()
            pbar = tqdm(opt.cln_trainloader, desc="Training Backdoor Generator")
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                cln_img = cln_img.to(opt.device)
                bd_gen_img = bd_gen(cln_img)
                cln_feat = inv_classifier.from_input_to_features(cln_img)
                bd_gen_feat = inv_classifier.from_input_to_features(bd_gen_img)
                loss_benign_feat = opt.mse(feat_mask*cln_feat, feat_mask*bd_gen_feat)
                loss_backdoor_feat = opt.mse((1-feat_mask)*cln_feat, (1-feat_mask)*bd_gen_feat)
                loss_norm = opt.mse(cln_img, bd_gen_img)

                if loss_norm > opt.norm_bound or loss_benign_feat > opt.feat_bound:
                    loss = loss_norm
                else:
                    loss = -loss_backdoor_feat + 0.01*loss_benign_feat
                    
                if n > 0:
                    inv_tlabel = torch.ones_like(targets, device=opt.device)*opt.detected_tlabel
                    bd_gen_pred = inv_classifier(bd_gen_img)
                    loss += opt.ce(bd_gen_pred, inv_tlabel)

                opt.opt_bd.zero_grad()
                loss.backward()
                opt.opt_bd.step()
                
                tloss += loss.item()
                tloss_benign_feat += loss_benign_feat.item()
                tloss_backdoor_feat += loss_backdoor_feat.item()
                tloss_norm += loss_norm.item()

                pbar.set_postfix({"round": "{:d}".format(n), 
                                "epoch": "{:d}".format(u),
                                "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                                "loss_bengin_feat": "{:.4f}".format(tloss_benign_feat/(batch_idx+1)),
                                "loss_backdoor_feat": "{:.4f}".format(tloss_backdoor_feat/(batch_idx+1)),
                                "loss_norm": "{:.4f}".format(tloss_norm/(batch_idx+1))})

    def unlearn(opt, model, bd_gen, n):
        classifier = model    
        for ul in range(opt.ul_round):
            tloss = 0
            tloss_pred = 0
            tloss_feat = 0
            bd_gen.eval()
            classifier.train()
            pbar = tqdm(opt.cln_trainloader, desc="Unlearning")
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                targets = targets.to(opt.device)
                bd_gen_num = int(0.1*cln_img.shape[0] + 1)
                bd_gen_list = random.sample(range(cln_img.shape[0]), bd_gen_num)
                cln_img = cln_img.to(opt.device)
                bd_gen_img = deepcopy(cln_img).to(opt.device)
                bd_gen_img[bd_gen_list] = bd_gen(bd_gen_img[bd_gen_list])

                cln_feat = classifier.from_input_to_features(cln_img)
                bd_gen_feat = classifier.from_input_to_features(bd_gen_img)
                bd_gen_pred = classifier.from_features_to_output(bd_gen_feat)
                loss_pred = opt.ce(bd_gen_pred, targets)
                loss_feat = opt.mse(cln_feat, bd_gen_feat)
                loss = loss_pred + loss_feat

                opt.opt_cls.zero_grad()
                loss.backward()
                opt.opt_cls.step()
            
                tloss += loss.item()
                tloss_pred += loss_pred.item()
                tloss_feat += loss_feat.item()
                pbar.set_postfix({"round": "{:d}".format(n), 
                                "epoch": "{:d}".format(ul),
                                "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                                "loss_pred": "{:.4f}".format(tloss_pred/(batch_idx+1)),
                                "loss_feat": "{:.4f}".format(tloss_feat/(batch_idx+1))})

            return classifier                            
            # if ((ul+1) % 10) == 0:
            #     opt.test(testloader=cln_testloader, testmodel=classifier, box=box, poisoned=False, poitarget=False, name="BA")
            #     test(testloader=cln_testloader, testmodel=classifier, box=box, poisoned=True, poitarget=True, passlabel=box.tlabel, name="ASR")
                    