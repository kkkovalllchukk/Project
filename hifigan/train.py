import argparse
import itertools
import json
import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from fastprogress import master_bar, progress_bar
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from .meldataset import (LogMelSpectrogram, MelDataset, get_dataset_filelist,
                         mel_spectrogram)
from .models import (Generator, MultiPeriodDiscriminator,
                     MultiScaleDiscriminator, discriminator_loss, feature_loss,
                     generator_loss)
from .utils import (AttrDict, build_env, load_checkpoint, plot_spectrogram,
                    save_checkpoint, scan_checkpoint)

torch.backends.cudnn.benchmark = True
USE_ALT_MELCALC = True


def train(rank, args, hyperparams):
    if hyperparams.num_gpus > 1:
        init_process_group(backend=hyperparams.dist_config['dist_backend'], init_method=hyperparams.dist_config['dist_url'],
                           world_size=hyperparams.dist_config['world_size'] * hyperparams.num_gpus, rank=rank)

    torch.cuda.manual_seed(hyperparams.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(hyperparams).to(device)
    multi_period_discriminator = MultiPeriodDiscriminator().to(device)
    multi_scale_discriminator = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(args.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", args.checkpoint_path)

    if os.path.isdir(args.checkpoint_path):
        cp_g = scan_checkpoint(args.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(args.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        multi_period_discriminator.load_state_dict(state_dict_do['mpd'])
        multi_scale_discriminator.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        print(f"Restored checkpoint from {cp_g} and {cp_do}")

    if hyperparams.num_gpus > 1:
        print("Multi-gpu detected")
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        multi_period_discriminator = DistributedDataParallel(multi_period_discriminator, device_ids=[rank]).to(device)
        multi_scale_discriminator = DistributedDataParallel(multi_scale_discriminator, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), hyperparams.learning_rate, betas=[hyperparams.adam_b1, hyperparams.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(multi_scale_discriminator.parameters(), multi_period_discriminator.parameters()),
                                hyperparams.learning_rate, betas=[hyperparams.adam_b1, hyperparams.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hyperparams.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hyperparams.lr_decay, last_epoch=last_epoch)
    if args.fp16:
        scaler_g = GradScaler()
        scaler_d = GradScaler()

    train_df, valid_df = get_dataset_filelist(args)

    trainset = MelDataset(train_df, hyperparams.segment_size, hyperparams.n_fft, hyperparams.num_mels,
                          hyperparams.hop_size, hyperparams.win_size, hyperparams.sampling_rate, hyperparams.fmin, hyperparams.fmax, n_cache_reuse=0,
                          shuffle=False if hyperparams.num_gpus > 1 else True, fmax_loss=hyperparams.fmax_for_loss, device=device,
                          fine_tuning=args.fine_tuning,
                          audio_root_path=args.audio_root_path, feat_root_path=args.feature_root_path, 
                          use_alt_melcalc=USE_ALT_MELCALC)

    train_sampler = DistributedSampler(trainset) if hyperparams.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=hyperparams.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=hyperparams.batch_size,
                              pin_memory=True,
                              persistent_workers=True,
                              drop_last=True)

    alt_melspec = LogMelSpectrogram(hyperparams.n_fft, hyperparams.num_mels, hyperparams.sampling_rate, hyperparams.hop_size, hyperparams.win_size, hyperparams.fmin, hyperparams.fmax).to(device)

    if rank == 0:
        validset = MelDataset(valid_df, hyperparams.segment_size, hyperparams.n_fft, hyperparams.num_mels,
                              hyperparams.hop_size, hyperparams.win_size, hyperparams.sampling_rate, hyperparams.fmin, hyperparams.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=hyperparams.fmax_for_loss, device=device, fine_tuning=args.fine_tuning,
                              audio_root_path=args.audio_root_path, feat_root_path=args.feature_root_path, 
                              use_alt_melcalc=USE_ALT_MELCALC)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       persistent_workers=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(args.checkpoint_path, 'logs'))

    generator.train()
    multi_period_discriminator.train()
    multi_scale_discriminator.train()
    
    if rank == 0: mb = master_bar(range(max(0, last_epoch), args.training_epochs))
    else: mb = range(max(0, last_epoch), args.training_epochs)

    for epoch in mb:
        if rank == 0:
            start = time.time()
            mb.write("Epoch: {}".format(epoch+1))

        if hyperparams.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        if rank == 0: pb = progress_bar(enumerate(train_loader), total=len(train_loader), parent=mb)
        else: pb = enumerate(train_loader)
        

        for i, batch in pb:
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)
            
            with torch.cuda.amp.autocast(enabled=args.fp16):
                y_g_hat = generator(x)
                if USE_ALT_MELCALC:
                    y_g_hat_mel = alt_melspec(y_g_hat.squeeze(1))
                else:
                    y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), hyperparams.n_fft, hyperparams.num_mels, hyperparams.sampling_rate, hyperparams.hop_size, hyperparams.win_size,
                                            hyperparams.fmin, hyperparams.fmax_for_loss)
            # print(x.shape, y_g_hat.shape, y_g_hat_mel.shape, y_mel.shape, y.shape)
            optim_d.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.fp16):
                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = multi_period_discriminator(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = multi_scale_discriminator(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f

            if args.fp16: 
                scaler_d.scale(loss_disc_all).backward()
                scaler_d.step(optim_d)
                scaler_d.update()
            else: 
                loss_disc_all.backward()
                optim_d.step()

            # Generator
            optim_g.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.fp16):
                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = multi_period_discriminator(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = multi_scale_discriminator(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all