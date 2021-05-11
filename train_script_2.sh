CUDA_VISIBLE_DEVICES=2,3 python trainval_cap_net.py \
                   --dataset clipart --net vgg16 \
                   --bs 12 \
                   --nw 6 \
                   --lr 0.001 \
                   --cuda \
                   --mGPUs \
                   --epochs 100 \
                   --save_model_dir clipart \
                   --lr_decay_step 40
