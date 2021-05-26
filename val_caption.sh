 CUDA_VISIBLE_DEVICES=3 python trainval_cap_net.py \
    --dataset coco_pascal_voc --net vgg16 \
    --bs 8 \
    --nw 4 \
    --cuda \
    --save_model_dir new_caption_model \
    --caption_for_da \
    --caption_ft_begin_iter 120000 \
    --caption_total_iter 200000\
    --cap_val_iter 10000 \
    --r true \
    --cap_resume true \
    --checksession 1 \
    --checkepoch 11 \
    --checkpoint 200000 


