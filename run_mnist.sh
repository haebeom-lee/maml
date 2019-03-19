python main.py \
  --savedir './results/mnist' \
  --dataset 'mnist' \
  --mode 'train' \
  --gpu_num 0 \
  --alpha 0.1 \
  --metabatch 10 \
  --n_steps 5 \
  --way 5 \
  --shot 1 \
  --query 5
  
  python main.py \
  --savedir './results/mnist' \
  --dataset 'mnist' \
  --mode 'test' \
  --gpu_num 0 \
  --alpha 0.1 \
  --metabatch 10 \
  --n_steps 5 \
  --way 5 \
  --shot 1 \
  --query 5

