cd ..
python train.py \
--backbone densenet121_scratch \
--train_dir '../datasets/covid19/train' \
--val_dir '../datasets/covid19/test' \
--num_classes 4 \
--batch_size 128 \
--max_epoch 160 \
--device cuda:0