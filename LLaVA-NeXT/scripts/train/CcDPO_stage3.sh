#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
echo "pythonpath="$PYTHONPATH

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

export WANDB_PROJECT=llava-OV-Multi-Image-TVDPO
export WANDB_NAME=T_DPO_type_1234

# DPO Stage
PROMPT_VERSION="qwen_1_5"
SFT_MODEL="CcDPO_stage2_pth"
EPOCH=1
beta=0.1

DPO_RUN_NAME="llava-onevision-qwen2-7b-ov-beta${beta}-epoch${EPOCH}_CcDPO_stage3"
DPO_CLEAN_NAME="${DPO_RUN_NAME##*/}"
OUTPUT_DIR="./ckpt/${DPO_CLEAN_NAME}"
DATA_PATH=“MultiScope-42k/stage3/stage3.json”

echo $DPO_RUN_NAME

#projector 和 llm可训练
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc-per-node=8 --master_port 31951 \
    llava/train/train_svco_triples.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path=${SFT_MODEL} \
    --dpo_alpha=1.0 \
    --beta=${beta} \
    --gamma=0.1 \
    --version $PROMPT_VERSION \
    --data_path=$DATA_PATH \
    --image_folder "MultiScope-42k/images/" \
    --unfreeze_mm_vision_tower False \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $DPO_CLEAN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 0.000001 \
    --weight_decay 0.02 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --use_image_type otherimage \
    --diffusion_step 500 \
    --dpo_token_weighted False \
    --dpo_token_weight 4.0 \
    --use_cross_modal_loss False \
    --use_symmetrical_loss True \
    --vdpo_weight 1.0 \
    --use_tdpo False \
    --tok_beta 0.1\
    --abs_weight 0.0


