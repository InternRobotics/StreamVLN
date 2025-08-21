#!/bin/bash
#SBATCH --job-name=blip3o    # Job name
#SBATCH --nodes=4                          # Number of nodes
#SBATCH --gres=gpu:8                         # Number of GPUs per node
#SBATCH --time=96:00:00                      # Time limit (hh:mm:ss)

export HF_HOME=/HF/Home/
MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20001))

VIDEO_FOLDER="data/trajectory_data/R2R","data/trajectory_data/RxR","data/dagger_data/R2R","data/dagger_data/RxR","data/dagger_data/EnvDrop"
MMC4_VIDEO_FOLDER="data/co-training_data/MMC4-core/images"
SCANQA_VIDEO_FOLDER="data/co-training_data/ScanNet"
QA_VIDEO_FOLDER="data/co-training_data/LLaVA-Video-178K"

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="StreamVLN_Video_${PROMPT_VERSION}_1epoch_196token_8history_32frame_stage_two"
PREV_STAGE_CHECKPOINT="StreamVLN_Video_${PROMPT_VERSION}_1epoch_196token_8history_32frame"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT streamvln/streamvln_train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --video_folder ${VIDEO_FOLDER} \
    --mmc4_video_folder ${MMC4_VIDEO_FOLDER} \
    --scanqa_video_folder ${SCANQA_VIDEO_FOLDER} \
    --qa_video_folder ${QA_VIDEO_FOLDER} \
    --group_by_task True \
    --multi_task_training True \
    \
    --num_history 8 \
    --num_future_steps 4 \
    --num_frames 32 \
    --data_augmentation True \
    --data_path "config/co-training_data.yaml" \
    \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --frames_upbound 32 \
    --force_sample True \
    --add_time_instruction True \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir checkpoints/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --mm_vision_tower_lr 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --report_to wandb \
