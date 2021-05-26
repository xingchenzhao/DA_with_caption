mkdir data/coco
mkdir data/coco/images
mkdir data/pretrained_model

wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm train2014.zip
mv train2014 data/coco/images

wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip
mv val2014 data/coco/images

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
rm annotations_trainval2014.zip
mv annotations data/coco

wget https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth
mv vgg16_caffe.pth data/pretrained_model