 CUDA_VISIBLE_DEVICES=2 python trainval_cap_net.py \
    --dataset coco_pascal_voc --net vgg16 \
    --bs 8 \
    --nw 4 \
    --cuda \
    --save_model_dir cap_model_scheduled_sampling \
    --caption_for_da \
    --caption_ft_begin_iter 120000 \
    --caption_total_iter 200000\
    --cap_val_iter 10000 \
    --wandb=da_cap \
    --wandb_id=img_caption_scheduled_sampleing_decay_fixed

