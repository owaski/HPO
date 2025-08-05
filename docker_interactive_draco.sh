ROOT=/lustre/fsw/portfolios/convai/users/souyang

CONTAINER_PATH=$ROOT/images/nemo_rl_npy.sqsh
CODE_DIR=$ROOT/code
CKPTS_DIR=$ROOT/ckpts
DATA_DIR=$ROOT/data
HF_CACHE_DIR=$ROOT/.cache/huggingface

NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)
HF_TOKEN=$(cat /home/souyang/.keys/hf_token)
WANDB_API_KEY=$(cat /home/souyang/.keys/wandb_api_key)

MOUNTS="/lustre/fs11:/lustre/fs11,${CODE_DIR}:/code,${CKPTS_DIR}:/ckpts,${DATA_DIR}:/data" \
CONTAINER=${CONTAINER_PATH} \
HF_DATASETS_CACHE=$HF_CACHE_DIR \
HF_TOKEN=${HF_TOKEN} \
WANDB_API_KEY=${WANDB_API_KEY} \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=llmservice_nemo_speechlm \
    --job-name=grpo-dev-infinisst \
    --partition=interactive \
    --time=2:0:0 \
    --mail-type=ALL \
    --mail-user=souyang@nvidia.com \
    ray.sub

# export NRL_VLLM_USE_V1=0
export PYTHONPATH=/code/RL:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
uv run python examples/run_grpo_infinisst.py \
    --config examples/configs/grpo_infinisst_interactive_vllm.yaml