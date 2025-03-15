cd ..
python train.py \
--backbone densenet121_finetune \
--train_dir '../datasets/covid19/train' \
--val_dir '../datasets/covid19/test' \
--num_classes 4 \
--batch_size 128 \
--max_epoch 160 \
--pretrain_model 'pretrain_model.ckpt' \
--device cuda:0