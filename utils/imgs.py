import cv2
import numpy as np


def crop(img, y, x, h, w):
    """random crop a single image with cropsize h, w at position y, x
    """
    return img[y:y + h, x:x + w, :]


def rescale(img, h, w, gray=False, nearest=False):
    """Image rescaler, handles gray input of (h, w, 1) and outputs gray as
    (h, w, 1) which opencv doesnt.
    nearest interpolation is needed for labels (classes shouldnt be smoothed)
    bilinear interpolation is used for everything else (smooth rescaling)
    """
    interpol = cv2.INTER_LINEAR
    if nearest:
        interpol = cv2.INTER_NEAREST
    if gray:
        img = np.squeeze(img)
    img = cv2.resize(img, (w, h), interpolation=interpol)
    if gray:
        img = np.expand_dims(img, -1)
    return img


def center_crop(img, h, w):
    """center crop a single image with cropsize x, y"""
    oh, ow, _ = img.shape
    dw = ow - w
    dh = oh - h
    startx = np.round(dw / 2).astype(int)
    starty = np.round(dh / 2).astype(int)
    return img[starty:starty + h, startx:startx + w, :]


def center_crop_batch(array, h, w):
    """center crop a batch of images with cropsize x, y"""
    _, oh, ow, _ = array.shape
    dw = ow - w
    dh = oh - h
    startx = np.round(dw / 2).astype(int)
    starty = np.round(dh / 2).astype(int)
    return array[:, starty:starty + h, startx:startx + w, :]


def test_crop():
    img = "D:\project_psp\datasets\city_small\images_singles\\test\\" \
          "berlin_000000_000019_leftImg8bit.png"
    arr = cv2.imread(img)
    arr = np.expand_dims(arr, 0)
    cropped = center_crop_batch(arr, 700, 500)[0]
    cv2.imshow('', cropped)
    cv2.waitKey(0)
    # cv2.imwrite("c:/test.png",)


def rescale_center_crop(arr, h, w, gray=False, nearest=False):
    """center crop over full image size (so crop only one dimension)
    and rescale to target size
    """
    oh, ow, _ = arr.shape
    rel_orig = oh / ow
    rel_target = h / w
    if rel_orig > rel_target:
        # crop vertically
        crop_w = ow
        crop_h = np.round(crop_w * rel_target).astype(int)
    else:
        # crop horizontally
        crop_h = oh
        crop_w = np.round(crop_h / rel_target).astype(int)
    # # do the crop
    # print("cropping orig", oh, ow, "target", h, w, "to", crop_h, crop_w)
    arr = center_crop(arr, crop_h, crop_w)
    # rescale to target size
    arr = rescale(arr, h, w, gray=gray, nearest=nearest)
    return arr


def test_val_crop():
    dir_ = "flow_test/crop_test"
    img = "datasets/city_small/images/test/" \
          "berlin_000000_000019_leftImg8bit.png"
    arr = cv2.imread(img)
    h, w = 100, 400
    cv2.imwrite(dir_ + "original.png", arr)
    arr = rescale_center_crop(arr, h, w)
    cv2.imwrite(dir_ + "cropped_scaled.png", arr)


if __name__ == '__main__':
    test_val_crop()
