CUDA_VISIBLE_DEVICES=4 \
python3 main.py \
--dataset cifar10 \
--arch resnet \
--depth 101 \
--epochs 160 \
-e_n resnet101_baseline