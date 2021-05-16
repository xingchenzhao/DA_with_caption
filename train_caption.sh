 CUDA_VISIBLE_DEVICES=3 python trainval_cap_net.py \
    --dataset coco_pascal_voc --net vgg16 \
    --bs 12 \
    --nw 4 \
    --cuda \
    --save_model_dir caption_model \
    --max_iter 30000 \
    --caption_for_da \
    --caption_ft_begin_iter 120000 \
    --caption_total_iter 200000\
    --cap_val_iter 5000

