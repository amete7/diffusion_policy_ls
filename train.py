import os
import torch
import torch.nn as nn
import numpy as np
import hydra
from dataset import CustomDataset, CustomDataset_Shared, init_dataset
import wandb
from tqdm.auto import tqdm
from model import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import time
from torch.utils.data import Subset

@hydra.main(config_path='./conf', config_name='config')
def main(cfg):
    if cfg.train.wandb:
        wandb.init(
            project=cfg.train.project_name,
            name='task_D_calvin_low_dim_check',
        )
    device = torch.device('cuda')

    dataset_folder_path = cfg.paths.dataset_folder_path
    stats_file_path = cfg.paths.stats_file_path
    processed_dataste_path = cfg.paths.processed_dataste_path

    # parameters
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
        custom_dataset = CustomDataset_Shared(data_dict, pred_horizon=pred_horizon, obs_horizon=obs_horizon)
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

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
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
    ema = EMAModel(
        model=noise_pred_net,
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

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    B = nobs.shape[0]

                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:,:obs_horizon,:]
                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(noise_pred_net)

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