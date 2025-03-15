cd ..
python train.py \
--backbone densenet121 \
--data_dir '../datasets/chestX-ray14/images' \
--num_classes 15 \
--batch_size 128 \
--max_epoch 160 \
--device cuda:0