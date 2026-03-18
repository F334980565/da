import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import Sampler
import wandb
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score, average_precision_score

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def make_weights_for_balanced_classes_split(dataset):
    num_classes = 4
    N = float(len(dataset))
    cls_ids = [[] for i in range(num_classes)]
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        cls_ids[label].append(idx)
    weight_per_class = [N / len(cls_ids[c]) for c in range(num_classes)]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        weight[idx] = weight_per_class[label]
    return torch.DoubleTensor(weight)


def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CV_Meter():
    def __init__(self, fold=5):
        self.fold = fold
        self.header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
        self.epochs = ["epoch"]
        self.cindex = ["cindex"]

    def updata(self, score, epoch):
        self.epochs.append(epoch)
        self.cindex.append(round(score, 4))

    def save(self, path):
        self.cindex.append(round(np.mean(self.cindex[1:self.fold + 1]), 4))
        self.cindex.append(round(np.std(self.cindex[1:self.fold + 1]), 4))
        print("save evaluation resluts to", path)
        with open(path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(self.header)
            writer.writerow(self.epochs)
            writer.writerow(self.cindex)
            
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, earliest_stop_epoch=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.earliest_stop_epoch = earliest_stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, score):
        
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.earliest_stop_epoch:
                self.early_stop = True

    def state_dict(self):
        return {
            'patience': self.patience,
            'earliest_stop_epoch': self.earliest_stop_epoch,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min
        }
        
    def load_state_dict(self,dict):
        self.patience = dict['patience']
        self.earliest_stop_epoch = dict['earliest_stop_epoch']
        self.verbose = dict['verbose']
        self.counter = dict['counter']
        self.best_score = dict['best_score']
        self.early_stop = dict['early_stop']
        self.val_loss_min = dict['val_loss_min']

class WandbLogger: 
    def __init__(self, args, fold=None):
        self.enabled = getattr(args, "wandb", False)
        self.args = args 
        self.fold = fold
        
        self.best_val_metric = None 
        self.best_epoch = None
    def log(self, metrics: dict, split: str = None, step: int | None = None,
            commit: bool = True):
        if not self.enabled or not metrics:
            return
        
        if split == "val":
            if self.best_val_metric is None or metrics['score'] > self.best_val_metric['score']:
                self.best_val_metric = metrics
                self.best_epoch = step

        rowd = {}
        split_prefix = f"{split}_" if split else ""

        for k, v in metrics.items():
            # 只 log 标量
            if isinstance(v, (int, float)):
                key = split_prefix + k
                rowd[key] = v

        if not rowd:
            return

        if step is not None:
            wandb.log(rowd, step=step, commit=commit)
        else:
            wandb.log(rowd, commit=commit)
            
    def print_epoch_summary(
        self,
        epoch: int,
        num_epoch: int,
        train_loss: float,
        val_metrics: dict,
        test_metrics=None,
        train_time_meter=None,
    ):
        pieces = [f"Fold {self.fold + 1}/{self.args.cv_fold}", f"Epoch [{epoch + 1}/{num_epoch}]"]
        for k, v in train_loss.items():
            pieces.append(f"{k}: {v:.3f}")

        val_loss = val_metrics.get("loss", None)
        if isinstance(val_loss, (int, float)):
            pieces.append(f"val loss: {val_loss:.4e}")

        for k, v in val_metrics.items():
            if k == "loss":
                continue
            if isinstance(v, (int, float)):
                pieces.append(f"val: {k}: {v:.3f}")
            else:
                pieces.append(f"val: {k}: {v}")
        
        if test_metrics is not None:
            for k, v in test_metrics.items():
                if isinstance(v, (int, float)):
                    pieces.append(f"test: {k}: {v:.3f}")
                else:
                    pieces.append(f"test: {k}: {v}")

        if train_time_meter is not None:
            pieces.append(
                f"time: {train_time_meter.val:.3f}({train_time_meter.avg:.3f})"
            )

        print(", ".join(pieces))
    
    def log_test_summary(
        self,
        test_metrics: dict,
        test_time=None,
        tag = 'test',
    ):
        pieces = []

        test_loss = test_metrics.get("loss", None)
        if isinstance(test_loss, (int, float)):
            pieces.append(f"{tag}_loss: {test_loss:.4e}")

        for k, v in test_metrics.items():
            if k == "loss":
                continue
            if isinstance(v, (int, float)):
                pieces.append(f"{k}: {v:.3f}")
            else:
                pieces.append(f"{k}: {v}")

        if test_time is not None:
            pieces.append(
                f"time: {test_time:.3f})"
            )
            
        print(", ".join(pieces))
        for k, v in test_metrics.items():
            k = f"{tag}_{k}"
            wandb.run.summary[k] = v
        
    def log_train_summary(self):
        pieces = []
        train_loss = self.best_val_metric.get("loss", None)
        if isinstance(train_loss, (int, float)):
            pieces.append(f"test loss: {train_loss:.4e}")

        for k, v in self.best_val_metric.items():
            if k == "loss":
                continue
            if isinstance(v, (int, float)):
                pieces.append(f"{k}: {v:.3f}")
            else:
                pieces.append(f"{k}: {v}")

        print(", ".join(pieces))
        
        for k in list(wandb.summary.keys()):
            if "test" in k:
                del wandb.summary[k]
        
        for k, v in self.best_val_metric.items():
            k = f"best_val_{k}"
            wandb.run.summary[k] = v

        wandb.run.summary[f"best_epoch"] = self.best_epoch
def patch_shuffle(x,group=0,g_idx=None,return_g_idx=False):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))

    # padding
    H, W = int(np.ceil(np.sqrt(p))), int(np.ceil(np.sqrt(p)))
    if group > H or group<= 0:
        return group_shuffle(x,group)
    _n = -H % group
    H, W = H+_n, W+_n
    add_length = H * W - p
    # print(add_length)
    ps = torch.cat([ps,torch.tensor([-1 for i in range(add_length)])])
    # patchify
    ps = ps.reshape(shape=(group,H//group,group,W//group))
    ps = torch.einsum('hpwq->hwpq',ps)
    ps = ps.reshape(shape=(group**2,H//group,W//group))
    # shuffle
    if g_idx is None:
        g_idx = torch.randperm(ps.size(0))
    ps = ps[g_idx]
    # unpatchify
    ps = ps.reshape(shape=(group,group,H//group,W//group))
    ps = torch.einsum('hwpq->hpwq',ps)
    ps = ps.reshape(shape=(H,W))
    idx = ps[ps>=0].view(p)
    
    if return_g_idx:
        return x[:,idx.long()],g_idx
    else:
        return x[:,idx.long()]

def group_shuffle(x,group=0):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))
    if group > 0 and group < p:
        _pad = -p % group
        ps = torch.cat([ps,torch.tensor([-1 for i in range(_pad)])])
        ps = ps.view(group,-1)
        g_idx = torch.randperm(ps.size(0))
        ps = ps[g_idx]
        idx = ps[ps>=0].view(p)
    else:
        idx = torch.randperm(p)
    return x[:,idx.long()]
def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    labels = np.array(dataset.slide_label)
    label_uni = set(dataset.slide_label)
    weight_per_class = [N/len(labels[labels==c]) for c in label_uni]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.slide_label[idx]
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

def cal_metrics(bag_labels, bag_predictions, sub_typing=False):
    bag_labels = np.array(bag_labels)
    bag_predictions = np.array(bag_predictions)

    if not sub_typing:
        if bag_predictions.ndim == 2 and bag_predictions.shape[1] > 1:
            score_for_nan = bag_predictions[:, 1]
        else:
            score_for_nan = bag_predictions.flatten()
    else:
        score_for_nan = np.max(bag_predictions, axis=1)

    mask = ~np.isnan(score_for_nan)

    bag_labels = bag_labels[mask]
    bag_predictions = bag_predictions[mask]

    if len(bag_labels) == 0:
        return 0, 0, np.nan, 0, 0, 0

    if len(np.unique(bag_labels)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
        y_pred = np.zeros_like(bag_labels)
        avg = 'binary' if not sub_typing else 'macro'
    else:
        try:
            if sub_typing:
                roc_auc = roc_auc_score(bag_labels, bag_predictions, multi_class='ovr')
                pr_auc = average_precision_score(bag_labels, bag_predictions)
                y_pred = np.argmax(bag_predictions, axis=1)
                avg = 'macro'
            else:
                pos_probs = bag_predictions[:, 1] if bag_predictions.ndim == 2 and bag_predictions.shape[1] > 1 else bag_predictions.flatten()

                fpr, tpr, threshold = roc_curve(bag_labels, pos_probs, pos_label=1)
                fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)

                roc_auc = roc_auc_score(bag_labels, pos_probs)
                pr_auc = average_precision_score(bag_labels, pos_probs)

                y_pred = (pos_probs >= threshold_optimal).astype(int)
                avg = 'binary'

        except ValueError as e:
            print(f"[WARN] Metric computation failed: {e}")
            return 0, 0, np.nan, 0, 0, 0

    precision, recall, fscore, _ = precision_recall_fscore_support(
        bag_labels, y_pred, average=avg, zero_division=0
    )
    accuracy = accuracy_score(bag_labels, y_pred)

    return accuracy, pr_auc, roc_auc, precision, recall, fscore

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule