from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from functools import partial

def get_split_dfs(args, df, is_test=False):
    if 'Split' not in df.columns:
        raise ValueError("CSV file must contain a 'Split' column")

    test_df = df[df['Split'].str.lower() == 'test'].reset_index(drop=True)
    train_df = df[df['Split'].str.lower() == 'train'].reset_index(drop=True)
    val_df = df[df['Split'].str.lower() == 'val'].reset_index(drop=True)
    if is_test:
        return test_df
    else:
        if len(val_df) == 0:
            print("Warning: val_df is empty, using test_df as val_df.")
            val_df = train_df
            
        return [train_df], [val_df], [test_df]

def get_patient_label(args, csv_file):
    print('[dataset] loading patient label from %s' % (csv_file))
    df = pd.read_csv(csv_file)
    required_columns = ['ID','Label']
    
    if not all(col in df.columns for col in required_columns):
        if len(df.columns) == 2:
            df.columns = ['ID', 'Label']
        elif len(df.columns) == 4:
            df.columns = ['Case', 'ID', 'Label', 'Split']
        else:
            df.columns = ['Case', 'ID', 'Label', 'DomainLabel', 'Split']
            
    patients_list = df['ID']
    labels_list = df['Label']
    
    label_counts = labels_list.value_counts().to_dict()
    print(f"patient_len:{len(patients_list)} label_len:{len(labels_list)}")
    print(f"all_counter:{label_counts}")
    
    return df

def get_patient_label_surv(args,csv_file):
	print('[dataset] loading dataset from %s' % (csv_file))
	rows = pd.read_csv(csv_file)
	rows = survival_label(rows)

	label_dist = rows['Label'].value_counts().sort_index()

	print('[dataset] discrete label distribution: ')
	print(label_dist)
	print('[dataset] dataset from %s, number of cases=%d' % (csv_file, len(rows)))

	return rows

def get_kfold(args,k, df, val_ratio=0, label_balance_val=True):
	skf = StratifiedKFold(n_splits=k)

	train_dfs = []
	val_dfs = []

	for train_index, val_index in skf.split(df, df['Label']):
		train_df = df.iloc[train_index]
		val_df = df.iloc[val_index]

		train_dfs.append(train_df)
		val_dfs.append(val_df)

	return train_dfs, val_dfs, val_dfs 

def survival_label(rows):
	n_bins, eps = 4, 1e-6
	uncensored_df = rows[rows['Status'] == 1]
	disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=n_bins, retbins=True, labels=False)
	q_bins[-1] = rows['Event'].max() + eps
	q_bins[0] = rows['Event'].min() - eps
	disc_labels, q_bins = pd.cut(rows['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
	# missing event data
	disc_labels = disc_labels.values.astype(int)
	disc_labels[disc_labels < 0] = -1
	if 'Label' not in rows.columns:
		rows.insert(len(rows.columns), 'Label', disc_labels)
	# Remove rows with label -1
	rows = rows[rows['Label'] != -1].reset_index(drop=True)
	return rows

def get_dataloader(args, dataset, df, split):

    args.num_domain = dataset.num_domain
    
    id2idx = {str(sid): i for i, sid in enumerate(dataset.slide_ids)}

    def df_to_indices(df):
        idxs = []
        for sid in df["ID"]:
            sid_str = str(sid)
            if sid_str in id2idx:
                idxs.append(id2idx[sid_str])
            else:
                continue

        return idxs

    indices = df_to_indices(df)
    print(f"[dataloader] {split} samples: {len(indices)}")

    subset = Subset(dataset, indices)

    pin_mem = torch.cuda.is_available()
    data_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=True,              
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    return data_loader