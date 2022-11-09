import os
train_imgs = "../../datasets/minne_apple/detection_full/train/images/"
test_imgs = "../../datasets/minne_apple/detection_full/test/images/"
train_masks = "../../datasets/minne_apple/detection_full/train/pixelwise_annotations/"
test_masks = "../../datasets/minne_apple/detection_full/test/pixelwise_annotations/"

train_imgs = [os.path.join(train_imgs, i) for i in os.listdir(train_imgs)]
train_masks = [os.path.join(train_masks, i) for i in os.listdir(train_masks)]

test_imgs = [os.path.join(test_imgs, i) for i in os.listdir(test_imgs)]
test_masks = [os.path.join(test_masks, i) for i in os.listdir(test_masks)]

generate_data = False
debug = False

n_debug = 10
n_segments = 700
n_color_channels = 3

"""
accuracy (GMM):0.6554723226929564
foreground accuracy (GMM):0.5138770281810419
background accuracy (GMM):0.6569838462660443

Evaluation on training data (pixels):
class_iou	0.03255203820836353
background_iou	0.6491475674508974
mean_iou	0.34084980282963045
accuracy	0.6532411412987148
class_accuracy	0.4421160122195643

Evaluation on test data (pixels):
class_iou	0.04333285008301749
background_iou	0.6406028459064367
mean_iou	0.34196784799472707
accuracy	0.6463597985376804
class_accuracy	0.517117527653662
"""