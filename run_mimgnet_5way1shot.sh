python main.py \
  --savedir './results/mimgnet_5way1shot' \
  --dataset 'mimgnet' \
  --mode 'train' \
  --gpu_num 0 \
  --alpha 0.01 \
  --metabatch 4 \
  --n_steps 5 \
  --way 5 \
  --shot 1 \
  --query 15 \
  --meta_lr 1e-4
  
  python main.py \
  --savedir './results/mimgnet_5way1shot' \
  --dataset 'mimgnet' \
  --mode 'test
  --gpu_num 0 \
  --alpha 0.01 \
  --metabatch 4 \
  --n_steps 10\
  --way 5 \
  --shot 1 \
  --query 15 \
  --meta_lr 1e-4
