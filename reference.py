import os
import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()

image_path = "/hdd/yhyun225/sora/videos/sora_video_first_frame/"
images = os.listdir(image_path)

generator = torch.manual_seed(42)

for image_name in images:
    image = Image.open(os.path.join(image_path, image_name))
    image = image.resize((1024, 576))
    frames = pipeline(image, num_frames=30, decode_chunk_size=8, generator=generator).frames[0]
    export_to_video(frames, "{}.mp4".format(image_name.strip('.png')), fps=7)