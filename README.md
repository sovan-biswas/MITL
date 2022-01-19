# MITL
Code for MITL - ICCVW 2021

The code for MITL is based on Pyslowfast code base. The current code is edited for MITL experiments. 

Please refer to [PySlowFast](https://github.com/facebookresearch/SlowFast) for dependencies and detailed usage.

### Config file
Please edit the following config file to reflect the path for the dataset and initial model. 
[SLOWFAST_32x2_R50_SHORT_cvg03_contrast.yaml](https://github.com/sovan-biswas/MITL/blob/master/configs/AVA/SLOWFAST_32x2_R50_SHORT_cvg03_contrast.yaml)

### Off-the-shelf Bounding box
The off-the-shelf bounding boxes are obtained by [Detectron](https://github.com/facebookresearch/detectron2) Resnext config ie. [X101-FPN	](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml)
