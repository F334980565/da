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
from utils.util import patch_shuffle, group_shuffle, cal_metrics
import torch.nn.functional as F
from torch.nn.functional import one_hot, cross_entropy
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score, average_precision_score
def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

class Engine(base_engine.BaseEngine):
    def __init__(self, args, results_dir, fold, task_type, logger = None):
        super().__init__(args, results_dir, fold, task_type, logger)
        self.lambda_cls = 1.0
        self.VIRTUAL_BATCH_SIZE = args.batch_size
        self.args = args
    def learning(self, model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()   
        
        device = next(model.parameters()).device
        start_epoch = 0 # 
        train_time_meter = AverageMeter()
        
        last_epoch = 0
        for epoch in range(start_epoch, self.args.num_epoch):
            start_t = time.time()
            train_loss = self.train(train_loader, test_loader, model, criterion, optimizer) 
            end_t = time.time()
            train_time_meter.update(end_t - start_t)
            val_metrics = self.validate(val_loader, model, criterion)
            if self.args.always_test:
                test_metrics = self.validate(test_loader, model, criterion)
            
            if self.logger is not None:
                self.logger.log(
                    train_loss,
                    split="train",
                    step=epoch,
                    commit=False,
                )
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
                        commit=True,
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
            
            if is_best or epoch % self.args.save_freq == 0:
                random_state = {
                    'np': np.random.get_state(),
                    'torch': torch.random.get_rng_state(),
                    'py': random.getstate(),
                }
                
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'k': self.fold, 
                    'early_stop': self.early_stopping.state_dict(),
                    'random': random_state,
                    'val_metrics': val_metrics,
                    'best_score': self.best_score,
                    'best_epoch': self.best_epoch,
                    'wandb_id': wandb.run.id if self.args.wandb else '',
                }
                
                self.save_checkpoint(
                    state, 
                    epoch, 
                    is_best, 
                    tag = "ckpt"
                )

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
        
        loss_cls_meter = AverageMeter()

        src_iter = infinite_loader(train_loader)
        tgt_iter = infinite_loader(test_loader)
        num_batches = len(train_loader)
        
        src_buffer = []
        tgt_buffer = []
        src_coords = []
        tgt_coords = []
        label_s_buffer = []
        label_t_buffer = []
        optimizer.zero_grad() 

        for batch_idx in range(num_batches):
            (_, _, src_feat, src_label, src_coord, _) = next(src_iter)
            (_, _, tgt_feat, tgt_label, tgt_coord, _) = next(tgt_iter)
            
            src_out = model.forward(src_feat.to(device), return_embed_ins_feat=False, return_encoded_ins_feat=False, return_attn=False, return_bag_feat=False)
            tgt_out = model.forward(tgt_feat.to(device), return_embed_ins_feat=False, return_encoded_ins_feat=False, return_attn=False, return_bag_feat=False)

            label_s_buffer.append(src_label.to(device))
            label_t_buffer.append(tgt_label.to(device))
            src_buffer.append(src_out)
            tgt_buffer.append(tgt_out)
            
            is_update_step = ((batch_idx + 1) % VIRTUAL_BATCH_SIZE == 0) or (batch_idx == num_batches - 1)
            if is_update_step:
                logits_t = torch.cat([x['logits'] for x in tgt_buffer], dim=0)
                logits_s = torch.cat([x['logits'] for x in src_buffer], dim=0)
                labels = torch.cat(label_s_buffer, dim=0)
                
                cls_loss = criterion(logits_s, labels)
                total_loss = self.lambda_cls * cls_loss
                
                total_loss.backward()
                
                loss_cls_meter.update(cls_loss.item())
                
                optimizer.step()
                optimizer.zero_grad()
                
                label_s_buffer = []
                label_t_buffer = []
                src_buffer = []
                tgt_buffer = []
                src_coords = []
                tgt_coords = []
                
            if batch_idx % self.args.log_iter == 0:
                print(f'[{batch_idx}/{num_batches}] cls:{loss_cls_meter.avg:.3f} ')

        return {'cls_loss': loss_cls_meter.avg}

