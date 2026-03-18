import argparse

def parse_args():
    # Dataset settings
    parser = argparse.ArgumentParser(description="configurations for response prediction")
    parser.add_argument('--dataset_name', default='camelyon16', type=str, help='[camelyon16, tcga, zlyy]')
    parser.add_argument('--feat_dir', type=str, help='提完的feature的目录 但不是最后一层 应该底下还有个pt_files')
    parser.add_argument('--csv_path', default=None, type=str, help='Dataset CSV path Label and Split')
    parser.add_argument('--h5_path', default=None, type=str, help='Dataset H5 path. Coord.')
    parser.add_argument('--num_classes', default=2, type=int, help='Label classes of dataset.')    
    # Early stop parameters
    parser.add_argument('--no_early_stop', action='store_true', help='因为我希望默认是early stop的')
    parser.add_argument('--patience', default=20, type=int, help='Label classes of dataset.')
    parser.add_argument('--earliest_stop_epoch', default=30, type=int, help='Min epoch to stop training.')
    
    # Training settings
    parser.add_argument('--cv_fold', default=1, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--always_test', action='store_true', help='Test model in the training phase')

    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducible experiment (default: 1)")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="如果只测试 就加上这个 这个测的一定是对应split为test的结果")
    parser.add_argument("--test_epoch", type=int, default=None, help="会自动从results_dir中对应存储的结果里选择测试的epoch， 为None的话会自动选择bestscore对应的epoch")
    
    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')
    
    # Model Parameters.
    parser.add_argument("--model", type=str, default="meanmil", help="type of model (default: meanmil)")
    parser.add_argument("--n_features", type=int, default=1024, help="dimension of input features.")
    parser.add_argument("--instance_dim", type=int, default=1024, help="dimension of input features.")
    parser.add_argument("--embed_instance_dim", type=int, default=1024, help="dimension of embedd features.") 
    parser.add_argument("--bag_dim", type=int, default=1024, help="dimension of pooled bag features.")
    
    # DA Parameters.
    parser.add_argument('--lambda_cls', default=1.0, type=float)
    parser.add_argument('--lambda_pca', default=0.1, type=float)
    parser.add_argument('--lambda_awpd', default=0.1, type=float)
    parser.add_argument('--lambda_energy', default=0.01, type=float)

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=30, help="maximum number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--loss", type=str, default="ce", help="slide-level classification loss function (default: ce)", choices=['ce', 'bce', 'ce_surv', 'nll_surv', 'nll_surv_l1', 'nll_surv_mse', 'nll_surv_kl', 'nll_surv_cos'])
    # Misc
    parser.add_argument('--title', default='default', type=str, help='Title of exp')
    parser.add_argument('--project', default='my_da_project', type=str, help='Project name of exp')
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--wandb_id', type=str, default=None, help='用于延续某个run? 或者需要手动指定run_id')
    parser.add_argument('--no_log', action='store_true', help='Without log')
    parser.add_argument('--model_path', type=str, help='Output path')
    parser.add_argument('--device', type=int, default=0, help='-1 means cpu') 
    parser.add_argument('--results_dir', type=str, default='./results', help='results dir for saving ckpt and wandb runs')
    parser.add_argument('--save_freq', type=int, default=20, help='多少epoch保存一次ckpt')
    
    # model parameters
    # RRTMIL
    parser.add_argument("--epeg_k", type=int, default=15, help="kernel size for epeg")
    parser.add_argument("--crmsa_k", type=int, default=3, help="kernel size for cr-msa")
    # Models with aux loss
    parser.add_argument("--cls_alpha", type=float, default=1.0, help="ratio for cls_loss")
    parser.add_argument("--aux_alpha", type=float, default=0.0, help="ratio for aux_loss")
    args = parser.parse_args()    
    
    # Dataset 
    # parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16, tcga]')
    # parser.add_argument('--dataset_root', default='/data/xxx/TransMIL', type=str, help='Dataset root path')
    # parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
    # parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    # parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    # parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    # parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]') 
    # parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    # parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    # parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')
    # parser.add_argument('--tcga_sub', default='nsclc', type=str, help='[nsclc,brca]')
    
    # # Train
    # parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
    # parser.add_argument('--aux_alpha', default=1.0, type=float, help='Auxiliary loss alpha')
    # parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    # parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
    # parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
    # parser.add_argument('--max_epoch', default=130, type=int, help='Number of max training epochs in the earlystopping [130]')
    # parser.add_argument('--input_dim', default=1024, type=int, help='dim of input features. PLIP features should be [512]')
    # parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    #parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    # parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    # parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce]')
    # parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    # parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    # parser.add_argument('--model', default='rrtmil', type=str, help='Model name')
    # parser.add_argument('--seed', default=2021, type=int, help='random number [2021]' )
    # parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    # parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    # parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    # parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    # parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    # parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
    # parser.add_argument('--set_path', type=str, default=None)

    # # Model
    # # Other models
    # parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # # Our
    # parser.add_argument('--only_rrt_enc',action='store_true', help='RRT+other MIL models [dsmil,clam,]')
    # parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    # parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    # # Transformer
    # parser.add_argument('--attn', default='rmsa', type=str, help='Inner attention')
    # parser.add_argument('--pool', default='attn', type=str, help='Classification poolinp. use abmil.')
    # parser.add_argument('--ffn', action='store_true', help='Feed-forward network. only for ablation')
    # parser.add_argument('--n_trans_layers', default=2, type=int, help='Number of layer in the transformer')
    # parser.add_argument('--mlp_ratio', default=4., type=int, help='Ratio of MLP in the FFN')
    # parser.add_argument('--qkv_bias', action='store_false')
    # parser.add_argument('--all_shortcut', action='store_true', help='x = x + rrt(x)')
    # # R-MSA
    # parser.add_argument('--region_attn', default='native', type=str, help='only for ablation')
    # parser.add_argument('--min_region_num', default=0, type=int, help='only for ablation')
    # parser.add_argument('--region_num', default=8, type=int, help='Number of the region. [8,12,16,...]')
    # parser.add_argument('--trans_dim', default=64, type=int, help='only for ablation')
    # parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the R-MSA')
    # parser.add_argument('--trans_drop_out', default=0.1, type=float, help='Dropout in the R-MSA')
    # parser.add_argument('--drop_path', default=0., type=float, help='Droppath in the R-MSA')
    # # PEG or PPEG. only for alation
    # parser.add_argument('--pos', default='none', type=str, help='Position embedding, enable PEG or PPEG')
    # parser.add_argument('--pos_pos', default=0, type=int, help='Position of pos embed [-1,0]')
    # parser.add_argument('--peg_k', default=7, type=int, help='K of the PEG and PPEG')
    # parser.add_argument('--peg_1d', action='store_true', help='1-D PEG and PPEG')
    # # EPEG
    # parser.add_argument('--epeg', action='store_false', help='enable epeg')
    # parser.add_argument('--epeg_bias', action='store_false', help='enable conv bias')
    # parser.add_argument('--epeg_2d', action='store_true', help='enable 2d conv. only for ablation')
    # parser.add_argument('--epeg_k', default=15, type=int, help='K of the EPEG. [9,15,21,...]')
    # parser.add_argument('--epeg_type', default='attn', type=str, help='only for ablation')
    # # CR-MSA
    # parser.add_argument('--cr_msa', action='store_false', help='enable CR-MSA')
    # parser.add_argument('--crmsa_k', default=3, type=int, help='K of the CR-MSA. [1,3,5]')
    # parser.add_argument('--crmsa_heads', default=8, type=int, help='head of CR-MSA. [1,8,...]')
    # parser.add_argument('--crmsa_mlp', action='store_true', help='mlp phi of CR-MSA?')

    # # DAttention
    # parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # # Shuffle
    # parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    # parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    # parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # # Misc
    # parser.add_argument('--title', default='default', type=str, help='Title of exp')
    # parser.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
    # parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    # parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    # parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    # parser.add_argument('--no_log', action='store_true', help='Without log')
    # parser.add_argument('--model_path', type=str, help='Output path')
    # parser.add_argument('--device', type=int, default=0, help='-1 means cpu')
    
    
    return args
