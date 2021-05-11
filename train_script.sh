CUDA_VISIBLE_DEVICES=2,3 python trainval_cap_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --bs 12 \
                   --nw 6 \
                   --lr 0.001 \
                   --cuda \
                   --mGPUs \
                   --epochs 13 \
                   --save_model_dir pascal_voc_no_lr_decay \
                   --lr_decay_step 20
