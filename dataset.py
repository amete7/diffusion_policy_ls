import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from utils.disk_dataset import DiskDataset
import time

def init_dataset(datasets_dir_path,pred_horizon):
    datasets_dir = Path(datasets_dir_path)
    key = "vis"  # Specify 'lang' or 'vis' depending on your dataset type
    lang_folder = "lang_annotations"  # Specify the language data folder name
    num_workers = 4  # Number of data loading workers
    transforms = {}  # Define any data transforms if needed
    batch_size = 32  # Specify your batch size
    min_window_size = pred_horizon  # Minimum window length
    max_window_size = pred_horizon  # Maximum window length
    pad = False  # Specify whether to pad sequences
    aux_lang_loss_window = 1  # Number of sliding windows for auxiliary language losses
    skip_frames = 1  # Number of frames to skip for language dataset
    save_format = "npz"  # File format in the dataset directory (either "pkl" or "npz")
    pretrain = False  # Set to True if you are pretraining
    obs_space = OmegaConf.create({'rgb_obs': [],
                'depth_obs': [],
                'state_obs': ['robot_obs', 'scene_obs'],
                'actions': ['actions'],
                'language': ['language']
    })
    proprio_state = OmegaConf.create({'n_state_obs': 54,
                    'keep_indices': [[0, 54]],
                    'robot_orientation_idx': [3, 6],
                    'normalize': True,
                    'normalize_robot_orientation': True
    })

    # Create a DiskDataset instance
    dataset = DiskDataset(
        datasets_dir=datasets_dir,
        obs_space=obs_space,
        proprio_state=proprio_state,
        key=key,
        lang_folder=lang_folder,
        num_workers=num_workers,
        transforms=transforms,
        batch_size=batch_size,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        pad=pad,
        aux_lang_loss_window=aux_lang_loss_window,
        skip_frames=skip_frames,
        save_format=save_format,
        pretrain=pretrain
    )
    return dataset

class CustomDataset(Dataset):
    def __init__(self, dataset, stats_info, pred_horizon=16, obs_horizon=2):
        self.dataset = dataset
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.init_min_max(stats_info)
    
    def init_min_max(self,stats_file_path):
        data = np.load(stats_file_path,allow_pickle=True)
        self.min_obs = data['min_obs']
        self.max_obs = data['max_obs']
        self.min_action = data['min_action']
        self.max_action = data['max_action']
    
    def normalize_data(self,obs,action):
        n_obs = 2 * (obs - self.min_obs) / (self.max_obs - self.min_obs) - 1
        n_action = 2 * (action - self.min_action) / (self.max_action - self.min_action) - 1
        return n_obs, n_action

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        print(time.time(),'started_data_load')
        sequence = self.dataset[idx, self.pred_horizon]
        obs = sequence['robot_obs']
        action = sequence['actions']
        print(time.time(),'loaded_obs_action')
        obs, action = self.normalize_data(obs,action)
        obs_sequence = obs[:self.obs_horizon]
        action_sequence = action[:self.pred_horizon]
        print(time.time(),'ended_data_load')
        return {'obs': obs_sequence, 'action': action_sequence}
    
class CustomDataset_Shared(Dataset):
    def __init__(self, dataset_dict, goals, pred_horizon=16, obs_horizon=2, isRelative = False):
        data_dict = dataset_dict
        self.obs = torch.from_numpy(data_dict['obs'])
        self.action = torch.from_numpy(data_dict['action'])
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.length = len(self.obs) - pred_horizon + 1
        self.goals = torch.from_numpy(goals['data'])
        self.action_dim = len(self.action[0])
        self.isRelative = isRelative

    def __len__(self):
        return self.length

    def find_nearest_idx(self, arr, target_idx):
        greater_indices = np.where(arr > target_idx)[0]  
        if len(greater_indices) == 0:
            # Last idx
            return arr[-1]
        return arr[greater_indices[0]]

    def __getitem__(self, idx):
        # print("Idx: ", idx)
        obs_sequence = self.obs[idx:idx + self.obs_horizon]
        # print("Obs_start, end: ", len(obs_sequence))
        action_sequence = self.action[idx:idx + self.pred_horizon]
        # Given current idx and 400 idx chunk
        goal_idx = self.find_nearest_idx(self.goals, idx)
        goal_ = self.obs[goal_idx]        
        goal = np.zeros_like(obs_sequence)
        goal[:] = goal_
        
        if  (goal_idx - idx) < self.pred_horizon:
            action_sequence = torch.zeros((self.pred_horizon, self.action_dim))
            action_sequence[: (goal_idx - idx)] = self.action[idx: goal_idx]
            if not self.isRelative:
                start = goal_idx - idx 
                pad_size = self.pred_horizon - (goal_idx - idx)
                for i in range(start, start + pad_size):
                    action_sequence[i, :] =  self.action[goal_idx-1]
                    
        return {'obs': obs_sequence, 'action': action_sequence, 'goal': goal}

    # def __getitem__(self, idx):
    #     obs_sequence = self.obs[idx:idx + self.obs_horizon]
    #     action_sequence = self.action[idx:idx + self.pred_horizon]
    #     print("action_start, end: ", idx, len(action_sequence))
    #     return {'obs': obs_sequence, 'action': action_sequence}
    