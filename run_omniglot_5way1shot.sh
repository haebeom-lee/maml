python main.py \
  --savedir './results/omni_5way1shot' \
  --dataset 'omniglot' \
  --mode 'train' \
  --gpu_num 0 \
  --alpha 0.4 \
  --metabatch 32 \
  --n_steps 1 \
  --way 5 \
  --shot 1 \
  --query 15 \
  --n_train_iters 40000

python main.py \
  --savedir './results/omni_5way1shot' \
  --dataset 'omniglot' \
  --mode 'test' \
  --gpu_num 0 \
  --alpha 0.4 \
  --metabatch 32 \
  --n_steps 1 \
  --way 5 \
  --shot 1 \
  --query 15 \
  --n_train_iters 40000
