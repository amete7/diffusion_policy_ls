from pathlib import Path
from omegaconf import OmegaConf
# from disk_dataset import DiskDataset
import numpy as np
import torch
import time
from tqdm import tqdm

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data):
    min_per_dim,_ = torch.min(data, dim=0)
    max_per_dim,_ = torch.max(data, dim=0)
    # print(min_per_dim)
    # print(max_per_dim)
    # nomalize to [0,1]
    ndata = 2 * (data - min_per_dim) / (max_per_dim - min_per_dim) - 1
    # normalize to [-1, 1]
    # ndata = ndata * 2 - 1
    return ndata, min_per_dim, max_per_dim

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def main():
    # # Define arguments for creating a DiskDataset instance
    # datasets_dir = Path("/satassdscratch/scml-shared/calvin_data/task_D_D/training")
    # # obs_space = {}  # Define your observation space as a dictionary
    # # proprio_state = {}  # Define your proprioceptive state as a dictionary
    # key = "vis"  # Specify 'lang' or 'vis' depending on your dataset type
    # lang_folder = "lang_annotations"  # Specify the language data folder name
    # num_workers = 4  # Number of data loading workers
    # transforms = {}  # Define any data transforms if needed
    # batch_size = 32  # Specify your batch size
    # min_window_size = 35  # Minimum window length
    # max_window_size = 35  # Maximum window length
    # pad = True  # Specify whether to pad sequences
    # aux_lang_loss_window = 1  # Number of sliding windows for auxiliary language losses
    # skip_frames = 1  # Number of frames to skip for language dataset
    # save_format = "npz"  # File format in the dataset directory (either "pkl" or "npz")
    # pretrain = False  # Set to True if you are pretraining
    # obs_space = OmegaConf.create({'rgb_obs': [],
    #             'depth_obs': [],
    #             'state_obs': ['robot_obs', 'scene_obs'],
    #             'actions': ['rel_actions'],
    #             'language': ['language']
    # })
    # proprio_state = OmegaConf.create({'n_state_obs': 54,
    #                 'keep_indices': [[0, 54]],
    #                 'robot_orientation_idx': [3, 6],
    #                 'normalize': True,
    #                 'normalize_robot_orientation': True
    # })

    # # Create a DiskDataset instance
    # dataset = DiskDataset(
    #     datasets_dir=datasets_dir,
    #     obs_space=obs_space,
    #     proprio_state=proprio_state,
    #     key=key,
    #     lang_folder=lang_folder,
    #     num_workers=num_workers,
    #     transforms=transforms,
    #     batch_size=batch_size,
    #     min_window_size=min_window_size,
    #     max_window_size=max_window_size,
    #     pad=pad,
    #     aux_lang_loss_window=aux_lang_loss_window,
    #     skip_frames=skip_frames,
    #     save_format=save_format,
    #     pretrain=pretrain
    # )
    
    # print("--- %s seconds for init---" % (time.time() - start_time))
    # init_time = time.time()
    # # for idx in range(len(dataset)):
    # # Load a specific data sequence (you can change idx to your desired index)
    # # Initialize global min and max tensors
    # dataset_length = len(dataset)
    # print("Dataset length:", dataset_length)
    
    # global_min_obs = torch.full((39,), float('inf'))
    # global_max_obs = torch.full((39,), float('-inf'))
    # global_min_action = torch.full((7,), float('inf'))
    # global_max_action = torch.full((7,), float('-inf'))

    # idx = 0  # Specify the index of the data sequence to load
    # cnt = 0
    # for idx in tqdm(range(0,len(dataset),min_window_size)):
    #     cnt+=1
    #     sequence = dataset.__getitem__([idx,max_window_size])
    #     # print("--- %s seconds for loading---" % (time.time() - init_time))
    #     # print("Loaded sequence:", sequence.keys())
    #     obs = sequence['robot_obs']
    #     # print(obs.type)
    #     # n_obs, min_obs, max_obs = normalize_data(obs)
    #     action = sequence['actions']

    # global_min_obs = torch.min(global_min_obs, min_obs)
    # global_max_obs = torch.max(global_max_obs, max_obs)
    global_min_action = [-0.432188, -0.545456,  0.293439, -3.141593, -0.811348, -3.141573, -1. ]
    global_max_action = [0.42977 , 0.139396, 0.796262, 3.141592, 0.638583, 3.141551, 1. ]
        # n_action, min_action, max_action = normalize_data(action)
        # nsample = {
        #     'obs': n_obs,
        #     'action': n_action
        # }
    stats = {
        # 'min_obs': global_min_obs,
        # 'max_obs': global_max_obs,
        'min_action': global_min_action,
        'max_action': global_max_action
    }
    # print(sample)
    # print(nsample)
    # n_sample = {key: value.numpy() for key, value in nsample.items()}
    # print(stats)
    # np.savez('/satassdscratch/scml-shared/calvin_data/calvin_debug_dataset/n_sample.npz', **nsample)
    np.savez('stats_actions.npz', **stats)
    # print(n_obs.shape)
        # Get the length of the dataset
    # print(cnt,'num_iter')
    print("--- %s seconds for completing---" % (time.time() - start_time))

if __name__ == "__main__":
    start_time = time.time()
    main()