CUDA_VISIBLE_DEVICES=5 \
python3 main.py \
--dataset cifar100 \
--arch resnet \
--depth 101 \
--epochs 160 \
-e_n resnet101_baseline