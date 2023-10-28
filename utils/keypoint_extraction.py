from tqdm import tqdm
import numpy as np
    

def _is_stopped(demo, i, obs, indices, delta=0.05):
    gripper_state_no_change = (
            i < (len(demo) - 1) and
            (obs[14] == demo[i + 1][14] and
             obs[14] == demo[i - 1][14]))
    
    # scene_states_unchanged = (i > (10) and i < (len(demo) - 10) and np.allclose(demo[i-1][15:], obs[15:], atol=delta) and np.allclose(obs[15:], demo[i+1][15:], atol=delta))

    # check to make sure none of the objects are moving
    scene_states_unchanged = (i > 1 and i < len(demo)-1 and np.allclose(demo[i-1][indices], obs[indices], atol=delta) and np.allclose(obs[indices], demo[i+1][indices], atol=delta))

    stopped = (scene_states_unchanged and gripper_state_no_change)
    return stopped


"""
    action_labels = ["x", "y", "z", "euler_x", "euler_y", "euler_z", "gripper"]

    state_obs_labels = ["sliding_door", "drawer", "button", "switch", "light_bulb", "green_light", "red_x", "red_y", "red_z", "red_euler_x", "red_euler_y", "red_euler_z",
                      "blue_x", "blue_y", "blue_z", "blue_euler_x", "blue_euler_y", "blue_euler_z", "pink_x", "pink_y", "pink_z", "pink_euler_x", "pink_euler_y", "pink_euler_z",]

    robot_obs_labels = ["x", "y", "z", "euler_x", "euler_y", "euler_z", "gripper_width", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]

    state_labels = robot_obs_labels + state_obs_labels
"""

def keypoint_discovery(demo, indices, delta=0.05):
    episode_keypoints = []
    # 14 corresponds to gripper, -1 is closed, 1 is open
    
    for i, obs in enumerate(tqdm(demo)):
        stopped = _is_stopped(demo, i, obs, indices, delta)
        
        # don't include first state as keypoint
        if not np.allclose(demo[0][indices], demo[i][indices], atol=0.01) and (stopped):
            # if len(episode_keypoints) == 0 or not np.allclose(demo[episode_keypoints[-1]][15:], demo[i][15:], atol=0.01):
            if len(episode_keypoints) == 0 or not np.allclose(demo[episode_keypoints[-1]][indices], demo[i][indices], atol=0.01):
                episode_keypoints.append(i)
    print('Found %d keypoints.' % len(episode_keypoints))
    return episode_keypoints

    
if __name__ == "__main__":
    file_path = "" # path to samples npz file
    data = np.load(file_path,allow_pickle=True)


    observations = data['obs']

    indices_wo_angles = [15, 16, 17, 18, 19, 20, 21, 22, 23, 27, 28, 29, 33, 34, 35] # provides better results, some angle information is a little noisy
    all_state_indices = np.arange(15, 39) # still works pretty well 
    keypoints = keypoint_discovery(observations, indices_wo_angles, 0.0001)