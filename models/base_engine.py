# models/base_engine.py
import os
import time
import torch
import random
import numpy as np
from abc import ABC
from tqdm import tqdm
from timm.utils import AverageMeter
from utils.util import WandbLogger, EarlyStopping
from utils.optimizer import define_optimizer
from sksurv.metrics import concordance_index_censored
from abc import ABC, abstractmethod
import wandb
from utils.util import patch_shuffle, group_shuffle, cal_metrics
from torch.nn.functional import one_hot, cross_entropy

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

class BaseEngine(ABC):
    def __init__(self, args, results_dir, fold, task_type='surv', logger = None): 
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        self.task_type = task_type 
        self.logger = logger

        self.best_score = 0.0
        self.best_epoch = 0
        self.filename_best = None
        self.early_stopping = EarlyStopping(
                                patience = args.patience,
                                earliest_stop_epoch = args.earliest_stop_epoch,
                                verbose = True,
                            )
        
    def learning(self, model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler):
                  
        if torch.cuda.is_available():
            model = model.cuda()   

        start_epoch = 0 # 
        train_time_meter = AverageMeter()
        
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
                        commit=True,
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
                    'lr_sche': scheduler.state_dict(),
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
        
        self.logger.log_train_summary()
        return self.best_score, self.best_epoch
    
    def testing(self, test_loader, model, criterion):
        fold_dir = os.path.join(self.results_dir, f'fold_{self.fold}')
        if self.args.test_epoch is None:
            ckp_filename = [file_name for file_name in os.listdir(fold_dir) if 'bestscore' in file_name][0]
        else:
            ckp_filename = f"ckp_epoch{self.args.test_epoch}.pth" 
            
        ckp_path = os.path.join(fold_dir, ckp_filename)
        if os.path.exists(ckp_path):
            ckp = torch.load(ckp_path, map_location="cpu")
        else:
            raise ValueError("No checkpoint found checkpoint path: {}".format(ckp_path))
        
        wandb_id = ckp.get("wandb_id", None)
        wandb.init(
            project=self.args.project,
            entity='d611',
            id=wandb_id,             
            resume="allow",          
        )
        model.load_state_dict(ckp["model"])
        if torch.cuda.is_available():
            model = model.cuda() 
        
        self.logger = WandbLogger(self.args, fold=self.fold)
        start_t = time.time()
        test_metrics = self.validate(test_loader, model, criterion)
        end_t = time.time()
        
        test_time = end_t - start_t
        self.logger.log_test_summary(test_metrics, test_time)
        return 
        
    def train(self, train_loader, test_loader, model, criterion, optimizer):
        if self.task_type == 'surv':
            return self._train_surv(train_loader, model, criterion, optimizer)
        elif self.task_type == 'cls':
            return self._train_cls(train_loader, model, criterion, optimizer)
        else:
            raise ValueError(f'Unknown task_type: {self.task_type}')

    def validate(self, data_loader, model, criterion):
        if self.task_type == 'surv':
            return self._val_surv(data_loader, model, criterion)
        elif self.task_type == 'cls':
            return self._val_cls(data_loader, model, criterion)
        else:
            raise ValueError(f'Unknown task_type: {self.task_type}')
    def _train_surv(self, data_loader, model, criterion, optimizer) -> dict:
        pass

    def _val_surv(self, data_loader, model, criterion) -> dict:
        pass
    
    def _train_cls(self, train_loader, model, criterion, optimizer):
        model.train()

        loss_total_meter = AverageMeter()
        loss_cl_meter  = AverageMeter()

        device = next(model.parameters()).device
        for batch_idx, (case_id, slide_id, feat, label, coord, domain_label_int) in enumerate(train_loader):
            optimizer.zero_grad()

            bag = feat.to(device)
            batch_size = bag.size(0)

            label = label.to(device)

            if self.args.patch_shuffle:
                bag = patch_shuffle(bag, self.args.shuffle_group)
            elif self.args.group_shuffle:
                bag = group_shuffle(bag, self.args.shuffle_group)

            result_dict = model(bag)
            train_logits = result_dict['logits']

            if self.args.loss == 'ce':
                cls_loss = criterion(train_logits.view(batch_size, -1), label)
            elif self.args.loss == 'bce':
                cls_loss = criterion(
                    train_logits.view(batch_size, -1),
                    one_hot(label.view(batch_size, -1), num_classes=self.args.n_classes).float()
                )
            else:
                raise ValueError(f"Unsupported loss type: {self.args.loss}")

            train_loss = self.args.cls_alpha * cls_loss

            train_loss.backward()
            optimizer.step()

            loss_total_meter.update(cls_loss.item(), 1)
            loss_cl_meter.update(float(cls_loss), 1)

            if batch_idx % self.args.log_iter == 0 or batch_idx == len(train_loader) - 1:
                if not getattr(self.args, 'no_log', False):
                    print(
                        f'[{batch_idx}/{len(train_loader)-1}] '
                        f'total_loss:{loss_total_meter.avg:.4f}, '
                        f'cls_loss:{loss_cl_meter.avg:.4f},  '
                    )

        return {'total_loss': loss_total_meter.avg}
    
    def _val_cls(self, data_loader, model, criterion):
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

    def save_checkpoint(self, state: dict, epoch: int, is_best: bool = False, tag: str = "ckpt"):
        ckpt_dir = os.path.join(self.results_dir, f'fold_{self.fold}')
        os.makedirs(ckpt_dir, exist_ok=True)

        if is_best:
            if hasattr(self, "best_path") and self.best_path is not None:
                if os.path.exists(self.best_path):
                    os.remove(self.best_path)
                    print(f"remove old BEST model: {self.best_path}")
            
            best_path = os.path.join(ckpt_dir, f"{tag}_model_bestscore_epoch{epoch:03d}.pth")
            torch.save(state, best_path)
            self.best_path = best_path
            print(f"save BEST model to {best_path}")
        else:
            ckpt_path = os.path.join(ckpt_dir, f"{tag}_epoch{epoch:03d}.pth")
            torch.save(state, ckpt_path)
            print(f"save checkpoint to {ckpt_path}")