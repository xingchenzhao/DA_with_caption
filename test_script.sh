CUDA_VISIBLE_DEVICES=5 python test_net.py --dataset clipart --net vgg16 \
                   --checksession 1 --checkepoch 2 --checkpoint 7000 \
                   --cuda --input_model caption_model_decay10_dw1