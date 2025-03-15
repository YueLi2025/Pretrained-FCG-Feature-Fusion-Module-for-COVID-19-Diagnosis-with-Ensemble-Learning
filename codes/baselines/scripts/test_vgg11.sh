cd ..
python train.py \
--backbone vgg11 \
--val_dir '../datasets/covid19/test' \
--checkpoint 'checkpoints/vgg11_best.ckpt' \
--num_classes 4 \
--batch_size 128 \
--device cuda:0 \
--eval