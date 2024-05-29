import json
import torch
import torch.nn as nn

def build_matrix(camera_extrinsic_params, relative_pose=False, zero_t_first_frame=False):
    b, f, d = camera_extrinsic_params.shape
    
    rx, ry, rz, tx, ty, tz = camera_extrinsic_params.chunk(6, dim=-1) # [b, f, 1] x 6
    Rx = torch.cat([
                torch.cos(rx), -torch.sin(rx), torch.zeros_like(rx),
                torch.sin(rx), torch.cos(rx), torch.zeros_like(rx),
                torch.zeros_like(rx), torch.zeros_like(rx), torch.ones_like(rx),
            ], dim=-1
        ).reshape(b, f, 3, 3) # [b, f, 3, 3]
    Ry = torch.cat([
                torch.cos(ry), torch.zeros_like(ry), torch.sin(ry),
                torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry),
                -torch.sin(ry), torch.zeros_like(ry), torch.cos(ry)
            ], dim=-1
        ).reshape(b, f, 3, 3) # [b, f, 3, 3]
    
    Rz = torch.cat([
            torch.ones_like(rz), torch.zeros_like(rz), torch.zeros_like(rz),
            torch.zeros_like(rz), torch.cos(rz), -torch.sin(rz),
            torch.zeros_like(rz), torch.sin(rz), torch.cos(rz)
            ], dim=-1
        ).reshape(b, f, 3, 3) # [b, f, 3, 3]
    R = Rx @ Ry @ Rz
    w2c_mat_4x4 = torch.eye(4).reshape(1, 1, 4, 4).expand(b, f, 4, 4) # [b, f, 4, 4]
    w2c_mat_4x4 = w2c_mat_4x4.to(device=R.device, dtype=R.dtype)
    w2c_mat_4x4[..., :3, :3] = R
    w2c_mat_4x4[..., 0, 3:] = tx
    w2c_mat_4x4[..., 1, 3:] = ty
    w2c_mat_4x4[..., 2, 3:] = tz
    
    w2c_mat = w2c_mat_4x4
    c2w_mat = torch.linalg.inv(w2c_mat_4x4)

    if relative_pose:
        c2w_mat = get_relative_pose(w2c_mat, c2w_mat, zero_t_first_frame=zero_t_first_frame)

    return w2c_mat, c2w_mat

def get_relative_pose(w2c, c2w, zero_t_first_frame=False):
    b, *_ = w2c.shape
    abs_w2cs = w2c                      # [b, f, 4, 4]
    abs_c2ws = c2w                      # [b, f, 4, 4]
    source_cam_c2w = abs_c2ws[:, 0]     # [b, 4, 4]

    if zero_t_first_frame:
        cam_to_origin = 0
    else:
        cam_to_origin = torch.linalg.norm(source_cam_c2w[:, :3, 3]) # [b, 1]
    
    target_cam_c2w = torch.eye(4).reshape(1, 4, 4).repeat(b, 1, 1)  # [b, 4, 4]
    target_cam_c2w = target_cam_c2w.to(device=c2w.device, dtype=c2w.dtype)
    target_cam_c2w[:, 1, -1:] = -cam_to_origin                      # [b, 4, 4]

    abs2rel = target_cam_c2w @ abs_w2cs[:, 0]                       # [b, 4, 4]
    abs2rel = abs2rel.unsqueeze(1)                                  # [b, 1, 4, 4]
    ret_poses = torch.cat(
        [target_cam_c2w.unsqueeze(1), abs2rel @ abs_c2ws[:, 1:, :, :]], dim=1
    )   # [b, f, 4, 4]
    
    return ret_poses

def get_plucker_embeddings(camera_params, sample_size, relative_pose=False, zero_t_first_frame=False):
    # K : intrinsic matrix [b, f, 4] - (fx, fy, px, py)
    # E : extrinsic matrix [b, f, 6] - (rx, ry, rz, tx, ty, tz)
    b, f, d = camera_params.shape
    if d == 6:
        # fixed camera intrinsics
        K = torch.tensor([0.5, 0.9, 0.5, 0.5]).reshape(1, 1, -1)
        K = K.repeat(b, f, 1)
        E = camera_params
    elif d == 10:
        K = camera_params[..., :4]
        E = camera_params[..., 4:]
    else:
        raise NotImplementedError
    
    H, W = sample_size
    K = K * torch.tensor((W, H)).repeat(2)  # [b, f, 4]
    fx, fy, px, py = K.chunk(4, dim=-1)     # [b, f, 1]

    j, i = torch.meshgrid(
        torch.linspace(0, H - 1, H),
        torch.linspace(0, W - 1, W),
        indexing='ij'
    )
    i = i.reshape([1, 1, H * W]).expand([b, f, H * W]) + 0.5   # [b, f, H x W]
    j = j.reshape([1, 1, H * W]).expand([b, f, H * W]) + 0.5   # [b, f, H x W]

    zs = torch.ones_like(i)     # [b, f, HxW]
    xs = (i - px) / fx * zs     # [b, f, HxW]
    ys = (j - py) / fy * zs     # [b, f, HxW]
    zs = zs.expand_as(ys)

    directions = torch.stack([xs, ys, zs], dim=-1)                  # [b, f, H x W, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True) # [b, f, H x W, 3]

    w2c, c2w = build_matrix(E, relative_pose=relative_pose, zero_t_first_frame=zero_t_first_frame)

    directions = directions.to(device=c2w.device)
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # [b, f, H x W, 3]
    rays_o = c2w[..., :3, 3]                                        # [b, f, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # [b, f, H x W, 3]

    rays_dxo = torch.cross(rays_o, rays_d)                          # [b, f, H x W, 3]
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)                 # [b, f, H x W, 6]
    plucker = plucker.reshape(b, f, H, W, 6)                        # [b, f, H, W, 6]
    plucker = plucker.permute(0, 1, 4, 2, 3).contiguous()           # [b, f, 6, H, W]

    return plucker


class LearnableCameraPose(object):
    def __init__(
            self, 
            num_data, 
            num_frames, 
            dof, 
            sample_size=(256, 384), 
            device="cuda", 
            dtype=torch.float32,
            relative_pose=False,
            zero_t_first_frame=False,
        ):
        self.num_data = num_data
        self.num_frames = num_frames
        self.dof = dof
        self.sample_size = sample_size
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        
        self.device = device
        self.dtype = dtype

        self.init_camera_pose()

    def init_camera_pose(self):
        camera_parameters = nn.Parameter(
            torch.zeros(
                (self.num_data, self.num_frames, self.dof), 
                device=self.device, 
                dtype=self.dtype
            )
        ).requires_grad_(True)

        self.camera_parameters = camera_parameters
    
    def get_poses(self, idx):
        return self.camera_parameters[idx]
    
    def plucker_embeddings(self, idx, order=None):
        camera_poses = self.get_poses(idx)          # [b, f, d]
        if order is not None:
            order = order.to(device=self.device)
            order = order.unsqueeze(-1).expand(-1, -1, camera_poses.shape[-1])
            camera_poses = torch.gather(camera_poses, 1, order)
    
        return get_plucker_embeddings(camera_poses, self.sample_size, relative_pose=self.relative_pose, zero_t_first_frame=self.zero_t_first_frame)
    
    def reverse_plucker_embeddings(self, idx, order=None):
        camera_poses = self.get_poses(idx)
        if order is not None:
            camera_poses = camera_poses[order]
        
        reverse_camera_poses = torch.flip(camera_poses, dims=[0])
        
        return get_plucker_embeddings(reverse_camera_poses, relative_pose=self.relative_pose, zero_t_first_frame=self.zero_t_first_frame)
        