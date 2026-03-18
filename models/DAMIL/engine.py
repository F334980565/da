import os
import torch
import time
import numpy as np
from tqdm import tqdm
from .. import base_engine 
from timm.utils import AverageMeter
from utils.util import WandbLogger, EarlyStopping
from utils.optimizer import define_optimizer
from sksurv.metrics import concordance_index_censored
from abc import ABC, abstractmethod
import wandb
import random
from utils.util import cal_metrics
import torch.nn.functional as F
from torch.nn.functional import one_hot, cross_entropy
from .network import get_pca_loss, get_awpd_loss, get_energy_loss

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

class Engine(base_engine.BaseEngine):
    def __init__(self, args, results_dir, fold, task_type, logger = None):
        super().__init__(args, results_dir, fold, task_type, logger) # 
        
        self.lambda_energy = args.lambda_energy
        self.lambda_pca = args.lambda_pca
        self.lambda_awpd = args.lambda_awpd
        self.lambda_cls = args.lambda_cls
        self.VIRTUAL_BATCH_SIZE = args.batch_size
        self.args = args
        
    def learning(self, model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()   
        
        device = next(model.parameters()).device
        start_epoch = 0 
        train_time_meter = AverageMeter()
        self.get_frozen_prototypes(train_loader, device)
        
        last_epoch = 0
        bank_size = min(512, len(train_loader.dataset), len(test_loader.dataset)) 
        model.set_bank_size(bank_size)
        model.warmup_iter = 5 * len(train_loader.dataset) // self.VIRTUAL_BATCH_SIZE
        for epoch in range(start_epoch, self.args.num_epoch):
            start_t = time.time()
            train_loss = self.train(train_loader, test_loader, model, criterion, optimizer)
            end_t = time.time()
            train_time_meter.update(end_t - start_t)
            val_metrics = self.validate(val_loader, model, criterion)
            if self.args.always_test:
                test_metrics = self.validate(test_loader, model, criterion)
            
            # wandb记录
            if self.logger is not None:
                # train 部分
                self.logger.log(
                    train_loss,
                    split="train",
                    step=epoch,
                    commit=False,
                )
                # val 部分
                if self.args.always_test:
                    self.logger.log(
                        val_metrics,
                        split="val",
                        step=epoch,
                        commit=False,
                    )
                    
                    self.logger.log(
                        test_metrics,
                        split="test",
                        step=epoch,
                        commit=False,
                    )                   
                else:
                    self.logger.log(
                        val_metrics,
                        split="val",
                        step=epoch,
                        commit=False,
                    )
                
                test_metrics = None if not self.args.always_test else test_metrics
                self.logger.print_epoch_summary(
                    epoch=epoch,
                    num_epoch=self.args.num_epoch,
                    train_loss=train_loss,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    train_time_meter=train_time_meter,
                )
            
            score = val_metrics['score']
            is_best = score > self.best_score
            if is_best:
                self.best_score = score
                self.best_epoch = epoch

            if scheduler is not None:
                scheduler.step()
            
            if not self.args.no_early_stop:
                self.early_stopping(epoch, score)
                if self.early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            last_epoch = epoch
        
        self.logger.log_train_summary()
        return self.best_score, self.best_epoch
    
    def train(self, train_loader, test_loader, model, criterion, optimizer): 
        model.train()
        device = next(model.parameters()).device
        VIRTUAL_BATCH_SIZE = self.VIRTUAL_BATCH_SIZE 
        
        meters = {
            'cls': AverageMeter(),
        }
        
        src_iter = infinite_loader(train_loader)
        tgt_iter = infinite_loader(test_loader)
        num_batches = len(train_loader)
        
        label_s_buffer, label_t_buffer = [], []
        src_buffer, tgt_buffer = [], []
        src_feats_buffer, tgt_feats_buffer = [], []
        
        optimizer.zero_grad() 

        for batch_idx in range(num_batches):
            (_, _, src_feat, src_label, _, _) = next(src_iter)
            (_, _, tgt_feat, tgt_label, _, _) = next(tgt_iter) 
            
            src_out = model(src_feat.to(device), return_bag_feat=True) 
            tgt_out = model(tgt_feat.to(device), return_bag_feat=True) 

            label_s_buffer.append(src_label.to(device))
            label_t_buffer.append(tgt_label.to(device)) 
            src_buffer.append(src_out)
            tgt_buffer.append(tgt_out)
            src_feats_buffer.append(src_feat.mean(dim=1))   # (B, D)
            tgt_feats_buffer.append(tgt_feat.mean(dim=1))
            
            is_update_step = ((batch_idx + 1) % VIRTUAL_BATCH_SIZE == 0) or (batch_idx == num_batches - 1)
            if is_update_step:
                logits_s = torch.cat([out['logits'] for out in src_buffer], dim=0)
                logits_t = torch.cat([out['logits'] for out in tgt_buffer], dim=0)
                labels_s = torch.cat(label_s_buffer, dim=0)
                labels_t = torch.cat(label_t_buffer, dim=0)
                src_feats = torch.cat(src_feats_buffer, dim=0)
                tgt_feats = torch.cat(tgt_feats_buffer, dim=0)
                
                ret_message = {}
                
                cls_loss = criterion(logits_s, labels_s)
                total_loss = self.lambda_cls * cls_loss
                meters['cls'].update(cls_loss.item())      
                    
                if self.lambda_pca > 0.0:
                    if not model.is_full:
                        proto_loss = torch.tensor(0.0)
                    else:
                        proto_loss = get_pca_loss(model, self.frozen_prototypes, self.args.num_classes, assign_mode='prototype')
                        total_loss += self.lambda_pca * proto_loss
                    
                if self.lambda_awpd > 0.0:
                    if not model.is_full:
                        proto_loss = torch.tensor(0.0)
                    else:
                        proto_loss = get_awpd_loss(model)
                        total_loss += self.lambda_awpd * proto_loss

                if self.lambda_energy > 0.0:
                    if not model.is_full or not model.current_iter > model.warmup_iter:
                        energy_loss = torch.tensor(0.0)
                    else:
                        energy_loss = get_energy_loss(model, self.frozen_prototypes)
                        total_loss += self.lambda_energy * energy_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                model.update_teacher()
                
                model.update_memory_bank(src_feats, torch.cat([out['bag_feat'] for out in src_buffer], dim=0), domain='src', labels=label_s_buffer)
                model.update_memory_bank(tgt_feats, torch.cat([out['bag_feat'] for out in tgt_buffer], dim=0), domain='tgt')
                
                for k, meter in meters.items():
                    if meter.count > 0:
                        ret_message[f'{k}_loss'] = meter.avg
                
                label_s_buffer, label_t_buffer = [], []
                src_buffer, tgt_buffer = [], []
                src_feats_buffer, tgt_feats_buffer = [], []

            if batch_idx % self.args.log_iter == 0 and 'ret_message' in locals():
                log_str = f'[{batch_idx}/{num_batches}] '
                log_str += " ".join([f"{k}:{v:.3f}" for k, v in ret_message.items()])
                print(log_str)

        return ret_message
    
    def get_frozen_prototypes(self, train_loader, device):
        all_feats = []
        all_labels = []
        
        with torch.no_grad():
            for batch in train_loader:
                (_, _, feat, label, _, _) = batch
                bag = feat.to(device)
                z = bag.mean(dim=1) 
                
                all_feats.append(z.cpu())
                all_labels.append(label.cpu())
        
        all_feats = torch.cat(all_feats, dim=0)   
        all_labels = torch.cat(all_labels, dim=0) 
        
        num_classes = len(torch.unique(all_labels))
        prototypes = []
        for i in range(num_classes):
            class_mask = (all_labels == i)
            if class_mask.any():
                class_proto = all_feats[class_mask].mean(dim=0)
                prototypes.append(class_proto)
            else:
                prototypes.append(torch.zeros(all_feats.shape[1]))
                
        self.frozen_prototypes = torch.stack(prototypes).to(device)
        
    def validate(self, data_loader, model, criterion):
        model.eval()
        loss_cls_meter = AverageMeter()
        test_loss_log = 0.
        bag_logit, bag_labels=[], []
        
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for i, (case_id, slide_id, feat, label, coord, domain_label) in enumerate(data_loader):
                bag_labels.append(label.item())
                bag = feat.to(device)
                label = label.to(device)
                result_dict = model(bag)
                test_logits = result_dict['logits']

                if self.args.loss == 'ce':
                    test_loss = criterion(test_logits, label)
                    probs = torch.softmax(test_logits, dim=-1).cpu().numpy() 
                    bag_logit.append(probs)
                elif self.args.loss == 'bce':
                    test_loss = criterion(test_logits, label.view(-1, 1).float())
                    probs = torch.sigmoid(test_logits).cpu().numpy()
                    bag_logit.append(probs)
                    
                loss_cls_meter.update(test_loss,1)
                
        bag_labels = np.array(bag_labels)
        bag_logit = np.concatenate(bag_logit, axis=0) 
        
        sub_typing = False
        if self.args.num_classes > 2:
            sub_typing = True
        
        accuracy, pr_auc, roc_auc, precision, recall, fscore = cal_metrics(bag_labels, bag_logit, sub_typing) 
        
        return {'score': roc_auc, 'roc_auc':roc_auc, 'pr_auc':pr_auc, 'acc':accuracy, 'precision':precision, 'recall':recall, 'fscore':fscore, 'loss':loss_cls_meter.avg}




