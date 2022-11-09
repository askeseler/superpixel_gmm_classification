import os
train_imgs = "../ProtoBasedSegmentation/datasets/wgisd_2/images/train"
test_imgs = "../ProtoBasedSegmentation/datasets/wgisd_2/images/test"
train_masks = "../ProtoBasedSegmentation/datasets/wgisd_2/pixelwise_annotations/train"
test_masks = "../ProtoBasedSegmentation/datasets/wgisd_2/pixelwise_annotations/test"

train_imgs = [os.path.join(train_imgs, i) for i in os.listdir(train_imgs)]
train_masks = [os.path.join(train_masks, i) for i in os.listdir(train_masks)]

test_imgs = [os.path.join(test_imgs, i) for i in os.listdir(test_imgs)]
test_masks = [os.path.join(test_masks, i) for i in os.listdir(test_masks)]

generate_data = False
debug = False

n_debug = 2
n_segments = 700
n_color_channels = 3


"""
accuracy (GMM):0.7197241419531683
foreground accuracy (GMM):0.6831157653134414
background accuracy (GMM):0.7238659417349671

Evaluation on training data (pixels):
class_iou	0.1985120846966097
background_iou	0.6866638623143422
mean_iou	0.44258797350547596
accuracy	0.7092296538097319
class_accuracy	0.6646617770746531

Evaluation on test data (pixels):
class_iou	0.2920504766097914
background_iou	0.732889095260404
mean_iou	0.5124697859350977
accuracy	0.7594010072200177
class_accuracy	0.7804081599488479
"""