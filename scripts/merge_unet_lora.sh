python tools/merge_lora2unet.py \
--lora_ckpt_path /home/yhyun225/CameraCtrl/ckpt/RealEstate10K_LoRA.ckpt \
--unet_ckpt_path runwayml/stable-diffusion-v1-5 \
--save_path /home/yhyun225/CameraCtrl/ckpt/unet_webvidlora_v3 \
--unet_config_path runwayml/stable-diffusion-v1-5/unet/config.json

# {
#     // IntelliSense를 사용하여 가능한 특성에 대해 알아보세요.
#     // 기존 특성에 대한 설명을 보려면 가리킵니다.
#     // 자세한 내용을 보려면 https://go.microsoft.com/fwlink/?linkid=830387을(를) 방문하세요.
#     "version": "0.2.0",   
#     "configurations": [
#         {
#             "name": "CameraCtrl",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "tools/merge_lora2unet.py",
#             "cwd": "${workspaceRoot}/CameraCtrl",
#             "console": "integratedTerminal",
#             "env": {"CUDA_VISIBLE_DEVICES": "0"},
#             "args": [
#                 "--lora_ckpt_path", "/home/yhyun225/CameraCtrl/ckpt/RealEstate10K_LoRA.ckpt",
#                 "--unet_ckpt_path", "runwayml/stable-diffusion-v1-5",
#                 "--save_path", "unet_webvidlora_v3",
#                 "--unet_config_path", "/home/yhyun225/CameraCtrl/configs/train_cameractrl/adv3_256_384_cameractrl_relora.yaml"
#             ]
#         }
#     ]
# }