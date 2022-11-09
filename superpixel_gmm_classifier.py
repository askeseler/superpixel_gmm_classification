import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import ImageCms
import sys
import importlib
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import shutil
from tqdm import tqdm
import gc
import random
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets._samples_generator import make_blobs
import os

def performance_metrics(balanced_labels, preds):
    with_acc_bg = (balanced_labels == 0)[preds == 0].sum()/np.sum(balanced_labels == 0)
    with_acc_fg = (balanced_labels == 1)[preds == 1].sum()/np.sum(balanced_labels == 1)
    acc = np.sum(balanced_labels == preds)/len(balanced_labels)
    return with_acc_bg, with_acc_fg, acc

def get_gmms(data, labels, n_components = 10):
    gmms = []
    for l in range(0, np.max(labels).astype(int) + 1):
        samples_of_class = data[labels == l]
        gmm = GaussianMixture(n_components=n_components, random_state=0).fit(samples_of_class)
        gmms.append(gmm)
    return gmms

def classify(gmms, data, bias = 0.00):
    likelyhoods = []
    
    for gm in gmms:
        likelyhood = -gm.score_samples(data)#np.exp(-gm.score_samples(data))
        likelyhoods.append(likelyhood)#log likelyhoods

    preds = 1 - np.argmax(likelyhoods, axis=0)
    #preds = (likelyhoods[0] / likelyhoods[1]) + bias > 1
    #preds = likelyhoods[0] > likelyhoods[1]
    return preds, np.array(likelyhoods)

from tqdm import tqdm
import gc
import random

def superpixel_colors_of_image(path, path_mask, w, h, tempdir = "temp", n_segments = 1000, sigma= 5, mode = "mean_colors"):
    hashval = random.getrandbits(128)
    hashval = "%032x" % hashval
    os.makedirs(os.path.join("tempfiles","superpixels", hashval), exist_ok = True)
    img = np.array(Image.open(path).resize((w,h)))
    mask = (np.array(Image.open(path_mask).resize((w,h))) > 0).astype(int)

    segments = slic(img, n_segments = n_segments, sigma = 5)
    memmap = np.memmap(os.path.join("tempfiles", "superpixels", os.path.basename(path).split(".")[0] + ".dat"), 
                       dtype='float32', shape = segments.shape, mode="w+")
    memmap[:] = segments[:]
    memmap.flush()
    
    n_pixels = np.max(segments)
    
    if mode == "mean_colors":
        superpixel_colors = [np.mean(img[segments == val], axis = 0) 
                                       for val in range(1, n_pixels)]
    elif mode == "dominant_colors":
        superpixel_colors = [dominant_color(img[segments == val]) 
                                       for val in range(1, n_pixels)]
        
    superpixel_labels = []
    for val in range(1, n_pixels):
        where = segments == val
        try:
            labels = mask[where].flatten()#binarize
            most_frequent_label = np.argmax(np.bincount(labels))
            #print(np.sum(labels) /len(labels))
            #print()
            superpixel_labels.append(most_frequent_label)
        except Exception as e:
            superpixel_labels.append(-1)
            print(np.sum(where))
            print(path_mask)
            print(e)
    del segments 
    gc.collect()
    return np.array(superpixel_colors, dtype=np.uint8), np.array(superpixel_labels, dtype=int), memmap

class ColorTrans:

    '''Class for transforming RGB<->LAB color spaces for PIL images.'''
    
    def __init__(self):
        self.srgb_p = ImageCms.createProfile("sRGB")
        self.lab_p  = ImageCms.createProfile("LAB")
        self.rgb2lab_trans = ImageCms.buildTransformFromOpenProfiles(self.srgb_p, self.lab_p, "RGB", "LAB")
        self.lab2rgb_trans = ImageCms.buildTransformFromOpenProfiles(self.lab_p, self.srgb_p, "LAB", "RGB")
    
    def rgb2lab(self, img):
        return ImageCms.applyTransform(img, self.rgb2lab_trans)

    def lab2rgb(self, img):
        return ImageCms.applyTransform(img, self.lab2rgb_trans)
lab_trans = ColorTrans()


def dominant_color(array, palette_size=32):
    pil_img = Image.fromarray(np.expand_dims(array, 0))
    # Resize image to speed up processing
    img = pil_img.copy()
    img.thumbnail((100, 100))

    # Reduce colors (uses k-means internally)
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=palette_size)

    # Find the color that occurs most often
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    palette_index = color_counts[0][1]
    dominant_color = palette[palette_index*3:palette_index*3+3]

    return dominant_color

def assign_to_image(superpixels, per_superpixel_label):
    out = np.zeros_like(superpixels).flatten()
    out[np.in1d(superpixels.flatten(), np.where(per_superpixel_label.astype(bool))[0] + 1).nonzero()[0]] = 1
    out = out.reshape(superpixels.shape)
    return out
        
def assign_to_images(gmms, superpixel_segmentations, superpixel_features):
    image_labels = []
    #superpixel_features = np.array_split(superpixel_features, len(superpixel_segmentations))
    for superpixels, superpixel_features_of_img in zip(tqdm(superpixel_segmentations), superpixel_features):
        #print(superpixel_features.shape)
        preds = classify(gmms, superpixel_features_of_img)[0]
        image_labels.append(assign_to_image(superpixels, preds))
    return np.array(image_labels)

def compute_segmentation_metrics(preds, targets):  
    where_true = targets == 1
    metrics = {}
    union_class = np.logical_or(preds == 1, targets == 1).astype(int).sum()
    intersection_class = np.logical_and(preds == 1, targets == 1).astype(int).sum()
    #metrics["class_iou"] = (preds[where_true] == 1).sum() / ((preds == 1).sum() + (preds[where_true] == 0).sum())
    metrics["class_iou"] = intersection_class / union_class
    metrics["background_iou"] = (preds[np.logical_not(where_true)] == 0).sum() / ((preds == 0).sum() + (preds[np.logical_not(where_true)] == 1).sum())
    metrics["mean_iou"] = (metrics["class_iou"] + metrics["background_iou"]) / 2
    metrics["accuracy"] = (preds == targets).sum() / (targets.shape[0] * targets.shape[1] * targets.shape[2])#correct classifications / all classifications
    metrics["class_accuracy"] = (preds[where_true] == 1).sum() / where_true.sum()#within foreground-class (where_true): pixels correctly classified as that class / all classifications of that class
    #(preds == 1).sum() + (preds[targets == 1] == 0) union of predictions == 1, and targets == 1 for the areas where predictions are not 1
    return metrics

def load_python_file(filepath):
    dir_name, file_name = os.path.split(cl_args.configuration_file)
    sys.path.insert(0, os.path.join(__file__, dir_name))
    args = importlib.import_module(os.path.splitext(file_name)[0])
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration_file", default="wgisd_2.py", type=str, help="path to configuration file")
    cl_args = parser.parse_args()
    args = load_python_file(cl_args.configuration_file)
    if not args.generate_data:
        if args.debug:
            train_imgs = args.train_imgs[:args.n_debug]
            test_imgs = args.test_imgs[:args.n_debug]
            train_masks = args.train_masks[:args.n_debug]
            test_masks = args.test_masks[:args.n_debug]
        else:
            train_imgs = args.train_imgs
            test_imgs = args.test_imgs
            train_masks = args.train_masks
            test_masks = args.test_masks
        w, h = Image.open(train_imgs[0]).size
        if os.path.isdir("tempfiles"):
            shutil.rmtree("tempfiles", ignore_errors=True)

        results = [superpixel_colors_of_image(path, path_mask, w, h, n_segments = args.n_segments) 
                for path, path_mask in zip(tqdm(train_imgs), train_masks)]

        colors_train_per_image = [matplotlib.colors.rgb_to_hsv(r[0])[:,:args.n_color_channels] for r in results]
        colors_train = np.concatenate(colors_train_per_image)
        #colors_train = matplotlib.colors.rgb_to_hsv(colors_train)[:,:args.n_color_channels]
        labels = np.concatenate([r[1] for r in results])
        memmaps_superpixels = [r[2] for r in results]
        #colors_train = matplotlib.colors.rgb_to_hsv(colors)[:,:args.n_color_channels]
        #lab = np.array(lab_trans.rgb2lab(Image.fromarray(np.expand_dims(colors_train, 0))))[0]
        balanced_labels = labels

    if args.generate_data:
        data, balanced_labels = make_blobs(n_samples=1000, centers=2, cluster_std=.5, random_state=0)
        data = data[:, ::-1]
        data = np.concatenate([data, data[balanced_labels]])#unbalanced classes
        balanced_labels = np.concatenate([balanced_labels,balanced_labels[balanced_labels]])
        data = data - np.min(data)
        data = data / np.max(data)
        labels = balanced_labels
        colors_train = data

    gmms = get_gmms(colors_train, balanced_labels)#fit on balanced data
    preds, likelyhoods = classify(gmms, colors_train, bias = 0.0)#classify all data

    with_acc_bg, with_acc_fg, acc = performance_metrics(labels, preds)

    print("accuracy (GMM):" + str(acc))
    print("foreground accuracy (GMM):" + str(with_acc_fg))
    print("background accuracy (GMM):" + str(with_acc_bg))

    if colors_train.shape[1] == 2:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from sklearn import mixture

        # display predicted scores by the model as a contour plot
        x = np.linspace(0, 1.0, 50)
        y = np.linspace(0, 1.0, 50)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = np.exp(-gmms[1].score_samples(XX))
        Z = Z.reshape(X.shape)

        Z1 = np.exp(-gmms[0].score_samples(XX))
        Z1 = Z1.reshape(X.shape)
        CS = plt.contour(
            X, Y, Z, levels=np.linspace(0,1,10)**3, color = "blue"
        )

        CS = plt.contour(
            X, Y, Z1, levels=np.linspace(0,1,10)**3, color = "blue"
        )

        plt.scatter(colors_train[balanced_labels == 0][:, 0], colors_train[balanced_labels == 0][:, 1], 0.8, c="blue")
        plt.scatter(colors_train[balanced_labels == 1][:, 0], colors_train[balanced_labels == 1][:, 1], 0.8, c ="red")

        plt.title("Negative log-likelihood predicted by a GMM")
        plt.axis("tight")
        plt.show()
    if colors_train.shape[1] == 2:
        fig, ax = plt.subplots(1,6, figsize=(12,2))
        ax[0].imshow(np.flipud(np.exp(-Z)))
        ax[1].imshow(np.flipud(np.exp(-Z1)))
        ax[2].imshow(np.flipud(np.exp(-Z)) - np.flipud(np.exp(-Z1)), cmap = "seismic")
        s = ax[3].imshow(np.flipud(np.exp(-Z)) *.95 - np.flipud(np.exp(-Z1))*.05, cmap = "seismic")
        for i, a in enumerate(ax):
            a.axis("off")
        fig.colorbar(s, ax = ax[4])
        plt.show()

    if not args.generate_data:
        predicted_pixels_train = assign_to_images(gmms, memmaps_superpixels, colors_train_per_image)
        masks_train = (np.array([np.array(Image.open(p).resize((w,h))) for p in train_masks]) > 0).astype(int)
        metrics_train = compute_segmentation_metrics(np.array(predicted_pixels_train).astype(int), masks_train)

        results = [superpixel_colors_of_image(path, path_mask, w, h, n_segments=args.n_segments) 
                        for path, path_mask in zip(tqdm(test_imgs), test_masks)]
    
        colors_test_per_image = [matplotlib.colors.rgb_to_hsv(r[0])[:,:args.n_color_channels]for r in results]
        colors_test = np.concatenate(colors_test_per_image)
        memmaps_superpixels_test = [r[2] for r in results]
        #colors_test = matplotlib.colors.rgb_to_hsv(colors_test)[:,:args.n_color_channels]
        #correct_classifications_test = np.concatenate([r[1] for r in results]) > 128#per superpixel
        #correct_classifications_test = np.array_split(correct_classifications_test, len(masks_train))

        predicted_pixels_test = assign_to_images(gmms, memmaps_superpixels_test, colors_test_per_image)
        masks_test = (np.array([np.array(Image.open(p).resize((w,h))) for p in test_masks]) > 0).astype(int)
        metrics_test = compute_segmentation_metrics(np.array(predicted_pixels_test).astype(int), masks_test)

        print("Evaluation on training data (pixels):")
        for k, v in metrics_train.items():
            print(k, end = "\t")
            print(v)

        print("Evaluation on test data (pixels):")
        for k, v in metrics_test.items():
            print(k, end = "\t")
            print(v)

        for i in range(args.n_debug):
            fig, ax = plt.subplots(1, 4, figsize = (10,15))
            ax[0].imshow(predicted_pixels_test[i])
            ax[1].imshow(np.array(Image.open(test_masks[i])) > 0)
            ax[2].imshow(Image.open(test_imgs[i]))
            ax[3].imshow(memmaps_superpixels_test[i], cmap = "rainbow")
            plt.show()