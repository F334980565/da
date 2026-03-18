import os
import re
import torch
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def get_label_int(dataset_name, labels):
	if dataset_name.lower().startswith("bio"):
		labels_int = [int(_l) for _l in labels]
	else:
		name = dataset_name.lower()
		if "panda" in name:
			labels_int = [int(_l) for _l in labels]
		elif "camelyon+" in name:
			labels_int = [0 if _l == "negative" else 1 for _l in labels]
		elif "her2" in name:
			labels_int = [0 if _l == "Negative" else 1 for _l in labels]
		else:
			raise NotImplementedError(
				f"Unknown dataset_name mapping rule: {dataset_name}"
			)
   
	return labels_int

class FeatClsDataset(Dataset):
    def __init__(self, dataset_name, feat_dir, label_path,
                 persistence=True, args=None):
        super(FeatClsDataset, self).__init__()

        self.dataset_name = dataset_name
        self.feat_dir = feat_dir
        self.label_path = label_path
        self.persistence = persistence
        h5_dir = args.h5_path if args is not None else None

        self.case_ids = []
        self.slide_ids = []
        self.feats = []   
        self.labels = []  
        self.coords = [] 
        self.domain_labels = []

        df = pd.read_csv(self.label_path)

        for _, row in df.iterrows():
            case_id = str(row["Case"])
            slide_id = str(row["ID"])
            label_str = str(row["Label"]) 
            domain_label = int(row.get("DomainLabel", -1)) 

            pt_path = os.path.join(self.feat_dir, "pt_files", f"{slide_id}.pt")
            if h5_dir is not None and os.path.exists(h5_dir):
                h5_path = os.path.join(h5_dir, f"{slide_id}.h5")
            else:
                h5_path = None

            if not os.path.exists(pt_path):
                continue

            self.case_ids.append(case_id)
            self.slide_ids.append(slide_id)
            self.labels.append(label_str)
            self.domain_labels.append(domain_label)

            if self.persistence:
                feat = torch.load(pt_path)  
                if h5_path is not None and os.path.exists(h5_path):
                    with h5py.File(h5_path, "r") as f:
                        coords = f["coords"][:] 
                else:
                    coords = np.zeros((feat.shape[0], 2), dtype=np.float32)
                self.feats.append(feat)
                self.coords.append(coords)
            else:
                self.feats.append(pt_path)
                self.coords.append(h5_path)
                
        self.num_domain = len(set(self.domain_labels))
        self.labels_int = get_label_int(self.dataset_name, self.labels)
        print(f"[dataset] {self.dataset_name} final size: {len(self.slide_ids)}")
        
    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        slide_id = self.slide_ids[idx]
        label_int = self.labels_int[idx]
        domain_label_int = self.domain_labels[idx]
        
        if self.persistence:
            feat = self.feats[idx]
            coord = self.coords[idx]
        else:
            feat = torch.load(self.feats[idx], weights_only=True)
            h5_coord_path = self.coords[idx]
            if h5_coord_path is not None and os.path.exists(h5_coord_path):
                with h5py.File(h5_coord_path, "r") as f:
                    coord = f["coords"][:]
            else:
                coord = np.zeros((feat.shape[0], 2), dtype=np.float32)

        return case_id, slide_id, feat, label_int, coord, domain_label_int

class FeatSurvDataset(Dataset):
	def __init__(self, df, root=None,persistence=True,keep_same_psize=0,is_train=False,return_id=False,args=None):
		self.root = os.path.join(root,'pt_files')
		# self.root = root
		self.persistence = persistence
		self.all_pts = os.listdir(self.root)
		self.keep_same_psize = keep_same_psize
		self.rows = df
		self.is_train = is_train
		self.return_id = return_id

		self.min_seq_len = args.min_seq_len if args else 10000000
		self.same_psize_pad_type = args.same_psize_pad_type if args else 'pad'
		self.h5_path = args.h5_path if args else None

		self.slide_name = {}
		for index, row in self.rows.iterrows():
			case_name = row['ID']
			if self.persistence:
				features = []
				patch_ids = []
				for slide_filename in self.all_pts:
					if case_name in slide_filename:
						feat = torch.load(os.path.join(self.root, slide_filename), weights_only=True)
						pid = [f"{slide_filename[:-3]}-{i}" for i in range(feat.shape[0])]
						features.append(feat)
						patch_ids.extend(pid)

				if len(features) > 0:
					features = torch.cat(features, dim=0)

					if self.keep_same_psize and self.is_train:
						features = get_same_psize(features,self.keep_same_psize,self.same_psize_pad_type,self.min_seq_len)

					self.slide_name[case_name] = (features, patch_ids)

				else:
					continue

			else:
				slides = [ slide for slide in self.all_pts if case_name in slide]
				
				if not slides:
					continue
				
				self.slide_name[str(case_name)] = slides
		
		self.rows = self.rows[self.rows['ID'].apply(lambda x: x in self.slide_name and bool(self.slide_name[x]))]
		self.rows.reset_index(drop=True, inplace=True)  
	
	def read_WSI(self, path):
		wsi = []
		all_patch_id = []
		for x in path:
			_wsi = torch.load(os.path.join(self.root,x),weights_only=True)
			wsi.append(_wsi)
			all_patch_id += [str(x)[:-3]+'-'+str(i) for i in range(_wsi.shape[0])]
		wsi = torch.cat(wsi, dim=0)
		if self.keep_same_psize and self.is_train:
			wsi = get_same_psize(wsi,self.keep_same_psize,self.same_psize_pad_type,self.min_seq_len)
		return wsi,all_patch_id

	def __getitem__(self, index):
		case = self.rows.loc[index, ['ID', 'Event', 'Status', 'Label']].values.tolist()
		ID, Event, Status, Label = case
		Censorship = 1 if int(Status) == 0 else 0
		if self.persistence:
			WSI, all_patch_id = self.slide_name[str(ID)]
		else:
			WSI,all_patch_id = self.read_WSI(self.slide_name[ID])
			
		_pos = None
		if self.h5_path is not None:
			if isinstance(self.slide_name[str(ID)], str):
				h5_file_stem = Path(self.slide_name[str(ID)]).stem
				h5_file_stems = []
			elif isinstance(self.slide_name[str(ID)], list):
				if len(self.slide_name[str(ID)]) == 1:
					h5_file_stem = Path(self.slide_name[str(ID)][0]).stem
					h5_file_stems = []
				else:
					h5_file_stems = [Path(slide).stem for slide in self.slide_name[str(ID)]]
					h5_file_stem = None
			else:
				h5_file_stem = None
				h5_file_stems = []

			if h5_file_stem is not None:
				pos_path = os.path.join(self.h5_path, h5_file_stem + '.h5')
				if os.path.isfile(pos_path):
					_pos = get_seq_pos_fn(pos_path)
					if self.keep_same_psize:
						if _pos is not None:
							WSI, _pos[1] = get_same_psize(WSI,
														  self.keep_same_psize,
														  self.same_psize_pad_type,
														  self.min_seq_len,
														  pos=_pos[1])
							new_max = _pos[1].max(dim=0)[0].unsqueeze(0)
							_pos[0] = new_max
						else:
							WSI = get_same_psize(WSI,
												 self.keep_same_psize,
												 self.same_psize_pad_type,
												 self.min_seq_len)
			elif len(h5_file_stems) > 1:
				all_coords = []
				for h5_file_stem in h5_file_stems:
					pos_path = os.path.join(self.h5_path, h5_file_stem + '.h5')
					if os.path.isfile(pos_path):
						current_pos = get_seq_pos_fn(pos_path)
						if current_pos is not None:
							all_coords.append(current_pos[1])

				if len(all_coords) > 0:
					merged_coords = torch.cat(all_coords, dim=0)
					max_coords = merged_coords.max(dim=0)[0].unsqueeze(0)
					_pos = [max_coords, merged_coords]

					if self.keep_same_psize:
						WSI, _pos[1] = get_same_psize(WSI,
													  self.keep_same_psize,
													  self.same_psize_pad_type,
													  self.min_seq_len,
													  pos=_pos[1])
						new_max = _pos[1].max(dim=0)[0].unsqueeze(0)
						_pos[0] = new_max
				else:
					_pos = None
					if self.keep_same_psize:
						WSI = get_same_psize(WSI,
											 self.keep_same_psize,
											 self.same_psize_pad_type,
											 self.min_seq_len)

		outputs = {
			'input': WSI,
			'event': Event,
			'censorship': Censorship,
			'target': Label 
		}

		if _pos is not None:
			_pos = torch.cat(_pos, dim=0)
			if (_pos.shape[0] - 1) != WSI.shape[0]:
				print(_pos.shape)
				print(WSI.shape)
				raise AssertionError("pos.shape 与特征.shape 不匹配")
			outputs['pos'] = _pos

		if self.return_id:
			outputs['idx'] = all_patch_id

		return outputs

	def __len__(self):
		return len(self.rows)

if __name__ == '__main__':
    pass