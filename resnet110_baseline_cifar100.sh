CUDA_VISIBLE_DEVICES=4 \
python3 main.py \
--dataset cifar100 \
--arch resnet \
--depth 110 \
--epochs 160 \
-e_n resnet110_baseline_cifar100