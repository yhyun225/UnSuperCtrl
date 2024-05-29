import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2

from decord import VideoReader
from torch.utils.data.dataset import Dataset
from packaging import version as pver
from tqdm import tqdm


class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)
    
class RealEstate10KPose(Dataset):
    def __init__(
            self,
            root_path,
            annotation_json,
            sample_stride=4,
            minimum_sample_stride=1,
            sample_n_frames=16,
            relative_pose=False,
            zero_t_first_frame=False,
            sample_size=(256, 384),
            shuffle_frames=False,
            use_flip=False,
            return_clip_name=False,
            **kwargs,
    ):
        self.root_path = root_path
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames
        self.return_clip_name = return_clip_name
        
        self.dataset = json.load(open(os.path.join(root_path, annotation_json), 'r'))[:10]

        self.sample_size = sample_size
        if use_flip:
            pixel_transforms = [
                transforms.Resize(sample_size),
                RandomHorizontalFlipWithPose(),
            ]
        else: 
            pixel_transforms = [
                transforms.Resize(sample_size, antialias=True),
            ]
        
        self.pixel_transforms = pixel_transforms
        self.shuffle_frames = shuffle_frames
        self.use_flip = use_flip

        self.prepare_frame_indices()
        

    def prepare_frame_indices(self):
        recons_dataset = []
        for i, video_dict in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            video_path = os.path.join(self.root_path, video_dict['clip_path'])
            total_frames = len(os.listdir(video_path))

            # videos with small number of frames
            if total_frames < self.sample_n_frames:
                continue

            current_sample_stride = self.sample_stride
            if total_frames < self.sample_n_frames * current_sample_stride:
                maximum_sample_stride = int(total_frames // self.sample_n_frames)
                current_sample_stride = random.randint(self.minimum_sample_stride, maximum_sample_stride)

            cropped_length = self.sample_n_frames * current_sample_stride
            start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
            end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

            assert end_frame_ind - start_frame_ind >= self.sample_n_frames
            frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)
            
            self.dataset[i]['indices'] = frame_indices
            video_dict['indices'] = frame_indices
            recons_dataset.append(video_dict)
        
        self.dataset = recons_dataset
        self.len = len(self.dataset)

    

    def load_video_dict(self, idx):
        video_dict = self.dataset[idx]

        video_path = os.path.join(self.root_path, video_dict['clip_path'])
        frames_path = sorted(os.listdir(video_path))

        frame_indices = video_dict['indices']
        
        frames = np.array([cv2.imread(os.path.join(video_path, frames_path[i])) for i in frame_indices])
        frames = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255. for frame in frames])
        frames = torch.from_numpy(frames).permute(0, 3, 1 ,2).contiguous()  # [f, c, h, w]

        first_frame_path = os.path.join(video_path, frames_path[frame_indices[0]])

        caption = video_dict['caption']
        clip_name = video_dict['clip_name']

        return frames, caption, clip_name, frame_indices, first_frame_path
    
    def get_batch(self, idx):
        (
            frames,         # [f, c , h, w]
            caption, 
            clip_name, 
            frame_indices,   # [f, ]
            first_frame_path,
        ) = self.load_video_dict(idx)
        
        perm = None
        if self.shuffle_frames:
            perm = np.random.permutation(self.sample_n_frames)
            frame_indices = frame_indices[perm]
            frames = frames[perm]

        frames = frames * 2.0 - 1.0
        
        reverse_frames = torch.flip(frames, dims=[0])

        return (
            first_frame_path,
            frames,
            reverse_frames, 
            caption,
            clip_name,
            perm
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # while True:
        #     try:
        #         (   
        #             image,
        #             video, 
        #             reverse_video, 
        #             caption,
        #             clip_name,
        #             order
        #         )  = self.get_batch(idx)
        #         break

        #     except Exception as e:
        #         idx = random.randint(0, self.len - 1)
        (   
            image_path,
            video, 
            reverse_video, 
            caption,
            clip_name,
            order
        )  = self.get_batch(idx)

        if self.use_flip:
            raise NotImplementedError
            # video = self.pixel_transforms[0](video)
            # video = self.pixel_transforms[1](video)
            # video = self.pixel_transforms[2](video)
        else:
            for transform in self.pixel_transforms:
                video = transform(video)
        
        sample = dict(
            image_paths=image_path,
            pixel_values=video,
            reverse_pixel_values=reverse_video,
            text=caption,
            idx=idx,
        )
        
        if self.return_clip_name:
            sample['clip_name'] = clip_name

        if order is not None:
            sample['order'] = order

        return sample

