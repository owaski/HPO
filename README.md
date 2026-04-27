# Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech

This is the repo for ACL 2026 paper "Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech". 
It only contains the RL post-training part. For SFT, please refer to [InfiniSST](https://github.com/LeiLiLab/InfiniSST) repo. 

## How to run the HPO training? 

We provide slurm script to run it on 3 8xH100 nodes. 
```bash
bash docker_sbatch_3node.sh YAML_NAME 
```
You can find the following yaml files under [examples/configs](examples/configs)

For En-Zh
- grpo_infinisst_4b_laal0.5_as_tgtq-5.0 (multiplier=1)
- grpo_infinisst_4b_laal0.5_as_tgtq-5.0_m{2...6} (multiplier=2...6)

For En-De
- grpo_infinisst_4b_laal0.5_as_tgtq-5.0_de (multiplier=1)
- grpo_infinisst_4b_laal0.5_as_tgtq-5.0_de_m{2...6} (multiplier=2...6)

For En-Ja
- grpo_infinisst_4b_laal0.5_as_tgtq-5.0_ja (multiplier=1)
- grpo_infinisst_4b_laal0.5_as_tgtq-5.0_ja_m{2...6} (multiplier=2...6)

Running this requires the SFT model checkpoint, the speech data pre-encoded into features with the speech encoder, and a docker container. 

We provide the example SFT checkpoint for en-zh [[here](https://drive.google.com/file/d/1m2Dd-hBqlFFSbAzVvaKuRYHK1om-DcJv/view?usp=sharing)], and the example pre-encoded data [[manifest](https://drive.google.com/file/d/1SonY83rcRP8TyugRacmrbzTMi5biC7et/view?usp=sharing)] [[encoded feature](https://drive.google.com/file/d/1L4QYMuFaQnCoYUqnq4PhL7X_wH2ecfZT/view?usp=sharing)] and the docker image [[here](https://drive.google.com/file/d/1xI7rT4pCuvNaA5rIFMUuuzUn4oeVgQxb/view?usp=sharing)]. 
