import math
import torch
import torch.nn as nn
from einops import rearrange

class PoseAdaptorPipeline(nn.Module):
    def __init__(self, unet, pose_encoder):
        super().__init__()
        self.unet = unet
        self.pose_encoder = pose_encoder

    def forward(
            self,
            noisy_latents, 
            timesteps, 
            added_time_ids,
            image_embeddings, 
            pose_embedding,     # [b, c, f, h, w] plucker embedding of the camera pose
        ):
        assert pose_embedding.ndim == 5
        b = pose_embedding.shape[0]
        pose_embedding_features = self.pose_encoder(pose_embedding)      # List[b x f, c, h, w]
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=b)
                                   for x in pose_embedding_features]
        
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            pose_embedding_features=pose_embedding_features,
            return_dict=False,
        )[0]

        return model_pred