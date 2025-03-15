cd ..
python test.py \
--backbone densenet121 \
--val_dir '../datasets/covid19/test' \
--checkpoint 'checkpoints/densenet121_mixup_best.ckpt' \
--num_classes 4 \
--batch_size 128 \
--device cuda:0 \
--eval