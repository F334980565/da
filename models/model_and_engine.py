import os
import wandb

from .AttMIL.network import DAttention as AMIL
from .AttMIL.engine import Engine as AttMIL_Engine

from .DAMIL.network import DAttMIL as DAMIL
from .DAMIL.engine import Engine as DAMILEngine

def build_model_and_engine(args, train_loader, criterion, results_dir, fold, task_type='surv', logger = None): #dtfd需要criterion 它直接在内部
    dataset_name = (
        args.excel_file.split("/")[-1].split(".")[0].split("_")[0]
        if hasattr(args, "excel_file") else "dataset"
    )
        
    if args.model == "AttMIL":
        model_cfg = dict(
            instance_dim = args.instance_dim,
            embed_instance_dim = args.embed_instance_dim,
            bag_dim = args.bag_dim,
            n_classes = args.num_classes,
            num_attention_heads = 1,
            attention_dim = 128,
            gated = False,
            bias = False,
            dropout = 0.25,
            act = "relu",
        )
        
        model = DAttention(**model_cfg)
        engine = AttMIL_Engine(args, results_dir, fold, task_type, logger)
    
    elif args.model == "DAMIL":
        model_cfg = dict(
            instance_dim = args.instance_dim,
            embed_instance_dim = args.embed_instance_dim,
            bag_dim = args.bag_dim,
            n_classes = args.num_classes,
            num_attention_heads = 1,
            attention_dim = 128,
            gated = False,
            bias = False,
            dropout = 0.25,
            act = "relu",
        ) 
        model = DAMIL(**model_cfg)
        engine = DAMILEngine(args, results_dir, fold, task_type, logger)
        
    else:
        raise NotImplementedError(f"model [{args.model}] is not implemented")
    
    if args.wandb and not args.evaluate:
        wandb.config.update(model_cfg)      
    
    return model, engine