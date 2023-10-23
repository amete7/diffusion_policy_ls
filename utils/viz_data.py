import numpy as np
import torch
# Load the .npy file
# file_path = "/home/atharva/roblr/calvin/n_sample.npz"
# file_path = "/satassdscratch/scml-shared/calvin_data/task_D_D/training/ep_start_end_ids.npy"  # Replace with the path to your .npy file
file_path = "/satassdscratch/scml-shared/calvin_data/task_D_D/stats.npz"
data = np.load(file_path,allow_pickle=True)

# Print the contents of the .npy file
# print("Contents of the .npy file:")
# nsample = {
#     'obs': torch.from_numpy(data['obs']),
#     'action': torch.from_numpy(data['action'])
# }
print(list(data.keys()))
print(data['min_obs'],data['max_obs'],data['min_action'],data['max_action'])
# print(nsample)
# print(np.shape(np.array(data['rgb_static'])))
# print("Contents of the .npy file:")
# print(data)

'''
each npz file contain one data point (one step)
total 2771 steps

'''
'''
# Load the .npy file
file_path = "/home/atharva/roblr/calvin/dataset/calvin_debug_dataset/training/ep_lens.npy"  # Replace with the path to your .npy file
data = np.load(file_path)

# Print the contents of the .npy file
print("Contents of the .npy file:")
print(data)
'''