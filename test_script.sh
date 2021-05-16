CUDA_VISIBLE_DEVICES=1 python test_net.py --dataset clipart --net vgg16 \
                   --checksession 1 --checkepoch 100 --checkpoint 82 \
                   --cuda --input_model clipart