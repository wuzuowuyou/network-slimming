CUDA_VISIBLE_DEVICES=4 \
python3 main.py \
--dataset cifar10 \
--arch vgg \
--depth 11 \
--epochs 50 \
-e_n test_tensorboard