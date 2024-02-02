# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
import os

wandb_log = False
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = int(os.environ.get('MODEL_BATCH_SIZE', 12))
block_size = int(os.environ.get('MODEL_BLOCK_SIZE', 512))
gradient_accumulation_steps = 5 * int(os.environ['WORLD_SIZE'])

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# baby model
n_layer = int(os.environ.get('MODEL_NUM_LAYERS', 12))
n_head = int(os.environ.get('MODEL_NUM_HEADS', 8))
n_embd = int(os.environ.get('MODEL_EMBEDDING_DIM', 512))
