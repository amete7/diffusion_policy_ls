# diffusion_policy_ls
Diffusion policy for Calvin Env

- We use hydra config manager, please check 'conf/config.yaml' for the parameters used
- The 'data' folder contains the following:
    - 'samples.npz' contains all the normalized datapoints from the task_D dataset
    - 'stats.npz' contains min-max values that can be used for de-normalization during inference.
- 'dataset.py' contains two classes, viz. CustomDatset & CustomDatset_Shared
    -use CustomDatset_Shared for loading the above 'samples.npz' file on memory
    use CustomDatset to directly load from original Calvin dataset
- 'model.py' contains diffusion policy model
- 'train.py' is the training script
- 'utils contains scripts used for data loading and data preprocessing