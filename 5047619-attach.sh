# No args launches on the head node (node 0)
# Args 1-N launch on worker nodes (nodes 1 through N-1)
WORKER_NUM=${1:-}
if [[ -z "$WORKER_NUM" ]]; then
  # Empty means we are on the head node
  srun --no-container-mount-home --gres=gpu:8 -A llmservice_nemo_reasoning -p interactive --overlap --container-name=ray-head --container-workdir=/lustre/fsw/portfolios/llmservice/users/souyang/code/RL-dev --nodes=1 --ntasks=1 -w "cw-dfw-h100-002-261-026" --jobid 5047619 --pty bash
else
  # Worker numbers 1 through N-1 correspond to ray-worker-1 through ray-worker-(N-1)
  # and use nodes_array[1] through nodes_array[N-1]
  if [[ $WORKER_NUM -lt 1 || $WORKER_NUM -ge 1 ]]; then
    echo "Error: WORKER_NUM must be between 1 and 0"
    exit 1
  fi
  nodes_array=(cw-dfw-h100-002-261-026)
  srun --no-container-mount-home --gres=gpu:8 -A llmservice_nemo_reasoning -p interactive --overlap --container-name=ray-worker-$WORKER_NUM --container-workdir=/lustre/fsw/portfolios/llmservice/users/souyang/code/RL-dev --nodes=1 --ntasks=1 -w "${nodes_array[$WORKER_NUM]}" --jobid 5047619 --pty bash
fi
