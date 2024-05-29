import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from diffusers import (
    AutoencoderKLTemporalDecoder,
    StableVideoDiffusionPipeline,
    EulerDiscreteScheduler
)
from diffusers.utils import load_image, export_to_video, BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from unsuperctrl.models.unet import UNetSpatioTemporalPoseConditionModel
from unsuperctrl.models.pose_encoder import CameraPoseEncoder
from einops import rearrange
  
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs

@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    frames: Union[List[PIL.Image.Image], np.ndarray]

class StableVideoDiffusionPoseControlPipeline(StableVideoDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalPoseConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
        pose_encoder: CameraPoseEncoder,
    ):
        super().__init__(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )

        self.register_modules(
            pose_encoder=pose_encoder
        )
    
    def train(self):
        self.unet.train()
        self.pose_encoder.train()
    
    def eval(self):
        self.unet.eval()
        self.pose_encoder.eval()
    
    def get_training_loss(
        self,
        pixel_values,   # Torch.Tensor[b, f, c, h, w] == [b, 16, 3, 256, 384]
        pose_embedding, # Torch.Tensor[b, c, f, h, w] == [b, 6, 16, 256, 384]
        t,              # Torch.Tensor[b]
        image,          # Torch.Tensor[b, c, h, w] == [b, 3, 256, 384]
        fps=7,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        num_videos_per_prompt=1,
        generator=None,
    ):
        # 0. Default tensor shapes & devices
        b, f, c, h, w = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype
        
        # 1. Encode input image: first frame of the video (for image embedding, crossattn)
        image_224 = F.interpolate(
            image, (224, 224), mode="bilinear", align_corners=False,
        )
        with torch.no_grad():
            image_embeddings = self._encode_image(image_224, device, num_videos_per_prompt, do_classifier_free_guidance=False)
            # [b, 1, 1024]
            
        fps = fps - 1

        # 2. Encode input image using VAE (concat to the latent model input)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        with torch.no_grad():
            image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance=False)
            image_latents = image_latents.to(image_embeddings.dtype) 
            # [b, c, h / 8, w / 8] = [b, 4, 32, 48]
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        image_latents = image_latents.unsqueeze(1).repeat(1, f, 1, 1, 1) # [b, f, 4, 32, 48]

        # 3. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            dtype,
            b,
            num_videos_per_prompt,
            do_classifier_free_guidance=False,
        )
        added_time_ids = added_time_ids.to(device) # [1, 3]

        with torch.no_grad():
            # 4. Encode training video
            pixel_values = rearrange(pixel_values, 'b f c h w -> (b f) c h w')  # [b x f, c, h, w]
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = rearrange(latents, "(b f) c h w -> b f c h w", b=b)       # [b, f, c, h, w] == [b, f, 4, 32, 48]
            latents = latents * self.vae.config.scaling_factor
        
        noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=dtype)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)  # [b, f, 4, 32, 48]

        # prepare unet input
        latent_model_input = torch.cat([noisy_latents, image_latents], dim=2) # [b, f, 8, 32, 48]

        # 7.5 Prepare pose embeddings [b, c, f, h, w] -> list([bf, c, h, w])
        # plucker embedding -> [pose encoder] -> camera pose feature
        pose_embedding_features = self.pose_encoder(pose_embedding)
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=b)
                                   for x in pose_embedding_features]
        
        model_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            pose_embedding_features=pose_embedding_features,
            return_dict=False
        )[0]

        if self.scheduler.config.prediction_type == "v_prediction":
            # [TODO] obtain GT 'v' with noisy latent & noise
            target = noise
        elif self.scheduler.config.prediction_type == "epsilon":
            target = noise
        else:
            raise NotImplementedError

        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        pose_embedding: Optional[torch.Tensor] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = "cuda" #self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale > 1.0

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width)
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 7.5 Prepare pose embeddings [b, c, f, h, w] -> list([bf, c, h, w])
        # plucker embedding -> [pose encoder] -> camera pose feature
        b = pose_embedding.shape[0]
        pose_embedding_features = self.pose_encoder(pose_embedding) if pose_embedding is not None else None
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=b)
                                   for x in pose_embedding_features]

        if do_classifier_free_guidance:
            pose_embedding_features = [torch.cat([x, x], dim=0) for x in pose_embedding_features]

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                    pose_embedding_features=pose_embedding_features,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)