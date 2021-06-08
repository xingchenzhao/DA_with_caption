CUDA_VISIBLE_DEVICES=7 python trainval_cap_net.py \
                   --cuda \
                   --dataset coco_pascal_voc \
                   --net vgg16 \
                   --lr 0.001 \
                   --max_iter 14000 \
                   --bs 12 \
                   --nw 4 \
                   --save_model_dir caption_model \
                   --caption_for_da \
                   --r true \
                   --cap_resume true \
                   --checksession 1 \
                   --checkepoch 4 \
                   --checkpoint 61560 \
                   --wandb da_cap \
                   --wandb_id da_bs12_dw0.1

