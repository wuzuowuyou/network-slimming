CUDA_VISIBLE_DEVICES=4 \
python3 main.py \
--dataset cifar10 \
--arch resnet \
--depth 56 \
--epochs 160 \
-e_n resnet56_baseline