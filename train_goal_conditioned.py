import os
import torch
import torch.nn as nn
import numpy as np
import hydra
from dataset import CustomDataset, CustomDataset_Shared, init_dataset
import wandb
from tqdm.auto import tqdm
from model.diffusion.conditional_unet1d import ConditionalUnet1D
from policy.diffusion_unet_lowdim_goal_policy import DiffusionUnetLowdimPolicy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.training_utils import EMAModel
from model.diffusion.ema_model import EMAModel
from diffusers.optimization import get_scheduler
import time
from torch.utils.data import Subset
import copy

@hydra.main(config_path='./conf', config_name='config')
def main(cfg):
    if cfg.train.wandb:
        wandb.init(
            project=cfg.train.project_name,
            name='task_D_calvin_low_dim_withGoals',
        )
    # device = torch.device('cuda')
    device = torch.device('cpu')

    dataset_folder_path = cfg.paths.dataset_folder_path
    stats_file_path = cfg.paths.stats_file_path
    processed_dataste_path = cfg.paths.processed_dataste_path
    goal_filepath = cfg.paths.goal_filepath

    # parameters
    # planning_horizon = cfg.params.planning_horizon
    pred_horizon = cfg.params.pred_horizon
    obs_horizon = cfg.params.obs_horizon
    action_horizon = cfg.params.action_horizon
    obs_dim = cfg.params.obs_dim
    action_dim = cfg.params.action_dim
    num_diffusion_iters = cfg.params.num_diffusion_iters
    
    save_interval = cfg.train.save_interval
    load_shared_data = cfg.train.load_shared_data
    num_epochs = cfg.train.num_epochs
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    if load_shared_data:
        data_dict = np.load(processed_dataste_path,allow_pickle=True)
        goals = np.load(goal_filepath)
        custom_dataset = CustomDataset_Shared(data_dict, goals, pred_horizon=pred_horizon, obs_horizon=obs_horizon)
    else:
        dataset = init_dataset(dataset_folder_path,pred_horizon)

        custom_dataset = CustomDataset(dataset, stats_file_path, pred_horizon=pred_horizon, obs_horizon=obs_horizon)

    print(len(custom_dataset),'number of training windows')

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        custom_dataset,
        batch_size=256,
        num_workers=1,
        shuffle=False,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    print('dataloader_loaded')
    # batch = next(iter(dataloader))
    # print("batch['obs'].shape:", batch['obs'].shape)
    # print("batch['action'].shape", batch['action'].shape)

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights

    # _target_: diffusion_policy.model.diffusion.conditional_unet1d.ConditionalUnet1D
    # input_dim: "${eval: ${task.action_dim} if ${obs_as_local_cond} or ${obs_as_global_cond} else ${task.obs_dim} + ${task.action_dim}}"
    # local_cond_dim: "${eval: ${task.obs_dim} if ${obs_as_local_cond} else None}"
    # global_cond_dim: "${eval: ${task.obs_dim}*${n_obs_steps} if ${obs_as_global_cond} else None}"
    # diffusion_step_embed_dim: 256
    # down_dims: [256, 512, 1024]
    # kernel_size: 5
    # n_groups: 8
    # cond_predict_scale: True
    
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon*2, #obs_dim*obs_horizon,
        diffusion_step_embed_dim=256,
        kernel_size=5,
        cond_predict_scale=True
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )
    _ = noise_pred_net.to(device)
    
    horizon =  16
    n_obs_steps = 2
    n_action_steps = 8
    n_latency_steps = 0
    past_action_visible = False
    keypoint_visible_rate = 1.0
    obs_as_local_cond = False
    obs_as_global_cond = True
    pred_action_steps_only = False

    policy = DiffusionUnetLowdimPolicy(model=noise_pred_net, noise_scheduler=noise_scheduler, 
                                       horizon=16, obs_dim=obs_dim, action_dim=action_dim, n_action_steps=n_action_steps,
                                       n_obs_steps=n_obs_steps, obs_as_global_cond=obs_as_global_cond)
    
    ema = EMAModel(
        model=copy.deepcopy(policy),
        power=cfg.train.ema_power)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=cfg.train.warmup_steps,
        num_training_steps=len(dataloader) * num_epochs
    )

    policy.model.train()
    
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                
                for batch_idx, nbatch in enumerate(tepoch):

                    # L2 loss
                    # loss = nn.functional.mse_loss(noise_pred, noise)
                    loss = policy.compute_loss(nbatch)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(policy)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            mean_epoch_loss = np.mean(epoch_loss)
            if cfg.train.wandb:
                wandb.log({"epoch_loss": mean_epoch_loss})
            if (epoch_idx+1) % save_interval == 0:
                print('saving checkpoint')
                ema_noise_pred_net = ema.averaged_model
                model_weights_path = cfg.paths.model_weights_path
                torch.save(ema_noise_pred_net.state_dict(), model_weights_path)

    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = ema.averaged_model

    model_weights_path = cfg.paths.model_weights_path
    torch.save(ema_noise_pred_net.state_dict(), model_weights_path)
    if cfg.train.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()