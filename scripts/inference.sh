# python -m torch.distributed.launch --nproc_per_node=8 --master_port=25000 inference.py \
#       --out_root ${OUTPUT_PATH} \
#       --ori_model_path ${SD1.5_PATH} \ 
#       --unet_subfolder ${SUBFOUDER_NAME} \
#       --motion_module_ckpt ${ADV3_MM_CKPT} \ 
#       --pose_adaptor_ckpt ${CAMERACTRL_CKPT} \
#       --model_config configs/train_cameractrl/adv3_256_384_cameractrl_relora.yaml \
#       --visualization_captions assets/cameractrl_prompts.json \
#       --use_specific_seeds \
#       --trajectory_file assets/pose_files/0f47577ab3441480.txt \
#       --n_procs 8

OUTPUT_PATH="outputs"
SD_PATH=ckpt/stable-diffusion-v1-5
SUBFOUDER_NAME=unet_webvidlora_v3
ADV3_MM_CKPT=ckpt/animatediff/v3_sd15_mm.ckpt
CAMERACTRL_CKPT=ckpt/cameractrl/CameraCtrl.ckpt

CUDA_VISIBLE_DEVICES=0 python inference.py \
      --out_root ${OUTPUT_PATH} \
      --ori_model_path $SD_PATH --unet_subfolder ${SUBFOUDER_NAME} \
      --motion_module_ckpt ${ADV3_MM_CKPT} --pose_adaptor_ckpt ${CAMERACTRL_CKPT} \
      --model_config configs/train_cameractrl/adv3_256_384_cameractrl_relora.yaml \
      --visualization_captions assets/cameractrl_prompts.json \
      --use_specific_seeds \
      --trajectory_file assets/pose_files/0f47577ab3441480.txt \