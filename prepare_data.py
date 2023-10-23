from pathlib import Path
from omegaconf import OmegaConf
from utils.disk_dataset import DiskDataset
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

def main():
    stats_file_path = "/satassdscratch/scml-shared/calvin_data/task_D_D/stats.npz"
    data = np.load(stats_file_path,allow_pickle=True)
    min_obs = data['min_obs']
    max_obs = data['max_obs']
    min_action = data['min_action']
    max_action = data['max_action']
    # Define arguments for creating a DiskDataset instance
    datasets_dir = Path("/satassdscratch/scml-shared/calvin_data/task_D_D/training")
    # obs_space = {}  # Define your observation space as a dictionary
    # proprio_state = {}  # Define your proprioceptive state as a dictionary
    key = "vis"  # Specify 'lang' or 'vis' depending on your dataset type
    lang_folder = "lang_annotations"  # Specify the language data folder name
    num_workers = 4  # Number of data loading workers
    transforms = {}  # Define any data transforms if needed
    batch_size = 32  # Specify your batch size
    min_window_size = 1  # Minimum window length
    max_window_size = 1  # Maximum window length
    pad = True  # Specify whether to pad sequences
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
    
    print("--- %s seconds for init---" % (time.time() - start_time))
    init_time = time.time()
    # for idx in range(len(dataset)):
    # Load a specific data sequence (you can change idx to your desired index)
    # Initialize global min and max tensors
    dataset_length = len(dataset)
    print("Dataset length:", dataset_length)
    idx = 0  # Specify the index of the data sequence to load
    cnt = 0
    observations = []
    actions = []
    for idx in tqdm(range(0,len(dataset))):
        cnt+=1
        sequence = dataset[idx, max_window_size]
        obs = sequence['robot_obs']
        action = sequence['actions']
        # print(time.time(),'loaded_obs_action')
        obs = 2 * (obs - min_obs) / (max_obs - min_obs) - 1
        action = 2 * (action - min_action) / (max_action - min_action) - 1
        # print(obs.shape,'obs_shape')
        # print(action.shape,'action_shape')
        # obs, action = normalize_data(obs,action)
        # obs_sequence = obs[:obs_horizon]
        # action_sequence = action[:pred_horizon]
        # print(time.time(),'ended_data_load')
        observations.append(obs.numpy()[0])
        actions.append(action.numpy()[0])
    # print(sample)
    # print(nsample)
    # n_sample = {key: value.numpy() for key, value in nsample.items()}
    # print(stats)
    observations = np.array(observations)
    actions = np.array(actions)
    print(observations)
    data = {'obs':observations, 'action': actions}
    # np.savez('/satassdscratch/scml-shared/calvin_data/calvin_debug_dataset/n_sample.npz', **nsample)
    np.savez('/satassdscratch/scml-shared/calvin_data/task_D_D/samples.npz', **data)
    # print(n_obs.shape)
        # Get the length of the dataset
    print(cnt,'num_iter')
    print("--- %s seconds for completing---" % (time.time() - start_time))

if __name__ == "__main__":
    start_time = time.time()
    main()