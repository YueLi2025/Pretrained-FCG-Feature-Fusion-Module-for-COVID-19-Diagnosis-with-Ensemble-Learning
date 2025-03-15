cd ..
python train.py \
--backbone densenet121 \
--val_dir '../datasets/covid19/test' \
--checkpoint 'checkpoints/densenet121_best.ckpt' \
--num_classes 4 \
--device cuda:0 \
--eval