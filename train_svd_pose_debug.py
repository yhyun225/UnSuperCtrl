import omegaconf.listconfig
import os
import math
import random
import time
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from diffusers import (
    AutoencoderKLTemporalDecoder, 
    EulerDiscreteScheduler,
    UNetSpatioTemporalConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.models.attention_processor import AttnProcessor

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from einops import rearrange

from unsuperctrl.data.dataset_no_pose import RealEstate10KPose
from unsuperctrl.data.camera_pose import LearnableCameraPose
from unsuperctrl.utils.util import setup_logger, format_time, save_videos_grid
from unsuperctrl.models.pipeline import StableVideoDiffusionPoseControlPipeline
from unsuperctrl.models.unet import UNetSpatioTemporalPoseConditionModel
from unsuperctrl.models.pipeline import PoseAdaptorPipeline
from unsuperctrl.models.pose_encoder import CameraPoseEncoder

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

def main(name: str,

         output_dir: str,
         pretrained_model_path: str,

         train_data: Dict,
         validation_data: Dict,
         cfg_random_null_text: bool = True,
         cfg_random_null_text_ratio: float = 0.1,

         unet_additional_kwargs: Dict = {},
         unet_subfolder: str = "unet",

         lora_rank: int = 4,
         lora_scale: float = 1.0,
         lora_ckpt: str = None,

         pose_encoder_kwargs: Dict = None,
         stable_video_diffusion_kwargs: Dict = None,

         do_sanity_check: bool = True,

         max_train_epoch: int = -1,
         max_train_steps: int = 100,
         validation_steps: int = 100,
         validation_steps_tuple: Tuple = (-1,),

         learning_rate: float = 3e-5,
         lr_warmup_steps: int = 0,
         lr_scheduler: str = "constant",

         num_workers: int = 32,
         train_batch_size: int = 1,
         adam_beta1: float = 0.9,
         adam_beta2: float = 0.999,
         adam_weight_decay: float = 1e-2,
         adam_epsilon: float = 1e-08,
         max_grad_norm: float = 1.0,
         gradient_accumulation_steps: int = 1,
         checkpointing_epochs: int = 5,
         checkpointing_steps: int = -1,

         mixed_precision_training: bool = True,

         global_seed: int = 42,
         logger_interval: int = 10,
         resume_from: str = None,

         camera_dof: int = 6
         ):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    # local_rank = init_dist(launcher=launcher, port=port)
    # global_rank = dist.get_rank()
    # num_processes = dist.get_world_size()
    # is_main_process = global_rank == 0

    global_rank = 0
    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)

    *_, config = inspect.getargvalues(inspect.currentframe())

    logger = setup_logger(output_dir, global_rank)

    # Handle the output folder creation
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    # OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    pretrained_model_name_or_path = stable_video_diffusion_kwargs.pretrained_model_name_or_path
    
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, subfolder="feature_extractor")
    
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    vae.eval()
    image_encoder.eval()

    # Load pose-conditional unet
    loading_kwargs = {
        "subfolder": "unet",
        "output_loading_info": True,
        "low_cpu_mem_usage": False,
    }
    unet, load_info_dict = UNetSpatioTemporalPoseConditionModel.from_pretrained(
        pretrained_model_name_or_path, **loading_kwargs
    )

    if lora_ckpt is not None:
        logger.info('image lora is not ready now')
    else:
        logger.info(f'We do not add image lora')

    missing_keys = load_info_dict["missing_keys"] # these keys correspond to linear projection layer

    for name, p in unet.named_parameters():
        if name in missing_keys:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    
    # load camera pose encoder
    pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)
    pose_encoder.requires_grad_(True)

    # Get the training dataset
    logger.info(f'Building training datasets')
    train_dataset = RealEstate10KPose(**train_data)
    logger.info(f'total {train_dataset.len} videos detected')

    # Get the validation dataset
    logger.info(f'Building validation datasets')
    validation_dataset = RealEstate10KPose(**validation_data)
    logger.info(f'total {validation_dataset.len} videos detected')

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Learnable Camera Pose creation:
    pose_loader = LearnableCameraPose(
        train_dataset.len,
        train_dataset.sample_n_frames,
        camera_dof,
        sample_size=train_data.sample_size,
        device="cuda",
        dtype=torch.float32,
        relative_pose=train_data.relative_pose,
        zero_t_first_frame=train_data.zero_t_first_frame,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    # Prepare svd pose-control pipeline -> for trainning & evaluation
    pipeline = StableVideoDiffusionPoseControlPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_encoder=pose_encoder,
    ).to("cuda")

    encoder_trainable_params = list(filter(lambda p: p.requires_grad, pipeline.pose_encoder.parameters()))
    encoder_trainable_param_names = [p[0] for p in
                                     filter(lambda p: p[1].requires_grad, pipeline.pose_encoder.named_parameters())]
    
    attention_trainable_params = [v for k, v in pipeline.unet.named_parameters() if v.requires_grad]
    attention_trainable_param_names = [k for k, v in pipeline.unet.named_parameters() if v.requires_grad]

    camera_trainable_params = [pose_loader.camera_parameters]
    camera_trainable_param_name = [f'learnable camera parameters']

    trainable_params = encoder_trainable_params + attention_trainable_params + camera_trainable_params
    trainable_param_names = encoder_trainable_param_names + attention_trainable_param_names + camera_trainable_param_name

    logger.info(f"trainable parameter names: {trainable_param_names}")
    logger.info(f"encoder trainable number: {sum(p.numel() for p in encoder_trainable_params) / 1e6:.3f} M")
    logger.info(f"attention processor trainable number: {sum(p.numel() for p in attention_trainable_params) / 1e6:.3f} M")
    logger.info(f"camera parameters trainable number: {sum(p.numel() for p in camera_trainable_params) / 1e6:.3f} M")
    logger.info(f"trainable parameter number: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")    

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    if resume_from is not None:
        logger.info(f"Resuming the training from the checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=pose_adaptor.device)
        global_step = ckpt['global_step']
        trained_iterations = (global_step % len(train_dataloader))
        first_epoch = int(global_step // len(train_dataloader))
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        pose_encoder_state_dict = ckpt['pose_encoder_state_dict']
        attention_processor_state_dict = ckpt['attention_processor_state_dict']
        pose_enc_m, pose_enc_u = pose_adaptor.module.pose_encoder.load_state_dict(pose_encoder_state_dict, strict=False)
        import pdb
        pdb.set_trace()
        assert len(pose_enc_m) == 0 and len(pose_enc_u) == 0
        _, attention_processor_u = pose_adaptor.module.unet.load_state_dict(attention_processor_state_dict, strict=False)
        assert len(attention_processor_u) == 0
        logger.info(f"Loading the pose encoder and attention processor weights done.")
        logger.info(f"Loading done, resuming training from the {global_step + 1}th iteration")
        lr_scheduler.last_epoch = first_epoch
    else:
        trained_iterations = 0

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None
    
    for epoch in range(first_epoch, num_train_epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        pipeline.train()

        data_iter = iter(train_dataloader)
        for step in range(trained_iterations, len(train_dataloader)):

            iter_start_time = time.time()
            batch = next(data_iter)
            data_end_time = time.time()

            # Data batch sanity check
            if epoch == first_epoch and step == 0 and do_sanity_check:
                pixel_values = batch['pixel_values'].cpu()                          # [b, f, c, h, w]
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")    # [b, c, f, h, w]
                for idx, pixel_value in enumerate(pixel_values):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value,
                                     f"{output_dir}/sanity_check/{global_rank}-{idx}.gif",
                                     rescale=True)

            ### >>>> Training >>>> ###

            # learnable camera poses
            idx = batch["idx"]
            frame_order = batch["order"] if train_data.shuffle_frames else None
            plucker_embedding = pose_loader.plucker_embeddings(idx, order=frame_order)  # [b, f, c(6), h(256), w(384)]

            # Convert videos to latent space
            pixel_values = batch["pixel_values"].to("cuda")     # [b, f, c, h, w]
            first_frames = batch["image"].to("cuda")            # [b, c, h, w]

            # Sample a random timestep for each video, [NOTE] confinuous timestep
            b = pixel_values.shape[0]
            t_idx = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,))
            timesteps = noise_scheduler.timesteps[t_idx]
            timesteps = timesteps.to("cuda")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            plucker_embedding = plucker_embedding.to(device="cuda")  # [b, f, 6, h, w]
            plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")  # [b, 6, f, h, w]
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                loss = pipeline.get_training_loss(
                    pixel_values=pixel_values,
                    pose_embedding=plucker_embedding,
                    t=timesteps,
                    image=first_frames,
                )

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, pipeline.parameters()),
                                               max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, pipeline.parameters()),
                                               max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            iter_end_time = time.time()

            # Save checkpoint
            if global_step % checkpointing_steps == 0:
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "pose_encoder_state_dict": pose_adaptor.pose_encoder.state_dict(),
                    "attention_processor_state_dict": {k: v for k, v in unet.state_dict().items()
                                                       if k in attention_trainable_param_names},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "camera_parameters": pose_loader.camera_parameters,
                }
                torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
                logger.info(f"Saved state to {save_path} (global_step: {global_step})")

            # Periodically validation
            if (global_step + 1) % validation_steps == 0 or (global_step + 1) in validation_steps_tuple:

                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)

                if isinstance(train_data, omegaconf.listconfig.ListConfig):
                    height = train_data[0].sample_size[0] if not isinstance(train_data[0].sample_size, int) else \
                    train_data[0].sample_size
                    width = train_data[0].sample_size[1] if not isinstance(train_data[0].sample_size, int) else \
                    train_data[0].sample_size
                else:
                    height = train_data.sample_size[0] if not isinstance(train_data.sample_size,
                                                                         int) else train_data.sample_size
                    width = train_data.sample_size[1] if not isinstance(train_data.sample_size,
                                                                        int) else train_data.sample_size

                validation_data_iter = iter(validation_dataloader)

                for idx, validation_batch in enumerate(validation_data_iter):
                    plucker_embedding = pose_loader.plucker_embeddings(validation_batch['idx'])
                    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")
                    sample = validation_pipeline(
                        prompt=validation_batch['text'],
                        pose_embedding=plucker_embedding,
                        video_length=video_length,
                        height=height,
                        width=width,
                        num_inference_steps=25,
                        guidance_scale=8.,
                        generator=generator,
                    ).videos[0]  # [3 f h w]
                    sample_gt = torch.cat([sample, (validation_batch['pixel_values'][0].permute(1, 0, 2, 3) + 1.0) / 2.0], dim=2)  # [3, f, 2h, w]
                    if 'clip_name' in validation_batch:
                        save_path = f"{output_dir}/samples/sample-{global_step}/{validation_batch['clip_name'][0]}.gif"
                    else:
                        save_path = f"{output_dir}/samples/sample-{global_step}/{idx}.gif"
                    save_videos_grid(sample_gt[None, ...], save_path)
                    logger.info(f"Saved samples to {save_path}")
            if (global_step % logger_interval) == 0 or global_step == 0:
                gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                msg = f"Iter: {global_step}/{max_train_steps}, Loss: {loss.detach().item(): .4f}, " \
                      f"lr: {lr_scheduler.get_last_lr()}, Data time: {format_time(data_end_time - iter_start_time)}, " \
                      f"Iter time: {format_time(iter_end_time - data_end_time)}, " \
                      f"ETA: {format_time((iter_end_time - iter_start_time) * (max_train_steps - global_step))}, " \
                      f"GPU memory: {gpu_memory: .2f} G"
                logger.info(msg)

            if global_step >= max_train_steps:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--camera_dof", type=int)
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, camera_dof=args.camera_dof, **config)
