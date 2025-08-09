ROOT=/lustre/fsw/portfolios/convai/users/souyang

CONTAINER_PATH=$ROOT/images/nemo_rl_npy.sqsh
CODE_DIR=$ROOT/code
CKPTS_DIR=$ROOT/ckpts
DATA_DIR=$ROOT/data
HF_CACHE_DIR=$ROOT/.cache/huggingface

NUM_ACTOR_NODES=3  # Total nodes requested (head is colocated on ray-worker-0)

CONFIG_NAME=$1
N=10

HF_TOKEN=$(cat /home/souyang/.keys/hf_token)
WANDB_API_KEY=$(cat /home/souyang/.keys/wandb_api_key)

JOB_NAME="${CONFIG_NAME}"

for i in $(seq 1 ${N}); do
    # COMMAND="NRL_VLLM_USE_V1=0 PYTHONPATH=/code/RL:$PYTHONPATH uv run ./examples/run_grpo_infinisst.py --config ${CONFIG_NAME}" \
    COMMAND="TOKENIZERS_PARALLELISM=false PYTHONPATH=/code/RL:$PYTHONPATH uv run ./examples/run_grpo_infinisst.py --config /code/RL/examples/configs/${CONFIG_NAME}.yaml" \
    MOUNTS="/lustre/fs11:/lustre/fs11,${CODE_DIR}:/code,${CKPTS_DIR}:/ckpts,${DATA_DIR}:/data" \
    CONTAINER=${CONTAINER_PATH} \
    HF_TOKEN=${HF_TOKEN} \
    HF_DATASETS_CACHE=$HF_CACHE_DIR \
    WANDB_API_KEY=${WANDB_API_KEY} \
    sbatch \
        --nodes=${NUM_ACTOR_NODES} \
        --account=llmservice_nemo_speechlm \
        --job-name=${JOB_NAME} \
        --partition=batch_block1,batch_block3,batch_block4 \
        --dependency=singleton \
        --time=4:0:0 \
        --mail-type=ALL \
        --mail-user=souyang@nvidia.com \
        ray.sub
done