import os
import time
import wandb
from mil_datasets.dataset import FeatClsDataset, FeatSurvDataset
from utils.options import parse_args
from utils.util import set_seed, WandbLogger
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from models.model_and_engine import build_model_and_engine
from mil_datasets.dataset_utils import get_patient_label, get_patient_label_surv, get_kfold, get_split_dfs, get_dataloader
from torch.utils.data import DataLoader, SubsetRandomSampler

def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    # create results directory
    
    results_dir = os.path.join(args.results_dir, args.title)
    fold_start = 0
    
    if not os.path.exists(results_dir):
        if args.evaluate:
            raise ValueError(f'test exp: {results_dir} does not exist')
        else:
            os.makedirs(results_dir)
    
    is_test = False
    if args.evaluate:
        is_test = True
    
    if args.dataset_name.lower().startswith('surv'):
        df = get_patient_label_surv(args, args.csv_path)
    else:
        df = get_patient_label(args, args.csv_path)
    
    if args.evaluate:
        test_dfs = get_split_dfs(args, df, is_test)
    else:
        if args.cv_fold > 1:
            train_dfs, val_dfs, test_dfs = get_kfold(args, args.cv_fold, df)
        else:
            train_dfs, val_dfs, test_dfs = get_split_dfs(args, df)
    
    if args.dataset_name.lower().startswith('surv'):
        dataset = FeatSurvDataset(excel_file=args.csv_path, folder=args.folder)
        task_type = 'surv'
    else:
        dataset = FeatClsDataset(args.dataset_name, args.feat_dir, args.csv_path, args.persistence, args)
        task_type = 'cls'
    
    if args.evaluate:
        k = 0
        test_loader = get_dataloader(args, dataset, test_dfs, split='test')
        criterion = define_loss(args)
        print("[model] loss function: ", args.loss)
        logger = WandbLogger(args, k)
        model, engine = build_model_and_engine(args, None, criterion, results_dir, k, task_type, logger)
        print("[model] trained model: ", args.model)        
        engine.testing(test_loader, model, criterion)
        wandb.finish()
    else:
        # start cv-fold CV evaluation.
        for k in range(fold_start, args.cv_fold):
            if args.wandb:
                if args.cv_fold > 1:
                    wandb.init(project=args.project)
                else:
                    wandb.init(project=args.project)
            
            train_loader = get_dataloader(args, dataset, train_dfs[k], split='train')
            val_loader = get_dataloader(args, dataset, val_dfs[k], split='val')
            test_loader = get_dataloader(args, dataset, test_dfs[k], split='test')

            # build model, criterion, optimizer, schedular
            criterion = define_loss(args)
            print("[model] loss function: ", args.loss)
            logger = WandbLogger(args, k)
            model, engine = build_model_and_engine(args, train_loader, criterion, results_dir, k, task_type, logger)
            print("[model] trained model: ", args.model)
            optimizer = define_optimizer(args, model)
            print("[model] optimizer: ", args.optimizer)
            scheduler = define_scheduler(args, optimizer)
            print("[model] scheduler: ", args.scheduler)
            # start training
            score, epoch = engine.learning(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler)
            wandb.finish()
            
if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
