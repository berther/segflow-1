import os
import cv2
import numpy as np

from utils.labels_info import labels
from keras.utils.np_utils import to_categorical


def main():
    labs = np.array([[4, 6, 1], [0, 5, 0], [7, 3, 2]])
    print("in", labs)
    onehots = encode_onehot(labs, 3, 3, 8)
    print("out", onehots.shape)
    print(onehots)
    labs_re = decode_onehot(onehots)
    print("re", labs_re)


def encode_onehot(labels_, h, w, n_labels):
    """one-hot encoding
    input classes as integers (h, w) or (h, w, 1)
    output one-hot encoding (h, w, n_classes)
    """
    if labels_.shape == (h, w, 1):
        labels_ = np.squeeze(labels_)
    # assert correct label size
    assert labels_.shape == (h, w), (
        "Labels where expected to be of shape ({}, {}) but are {}".format(
            h, w, labels_.shape))
    labels_ = np.reshape(labels_, (h * w))
    # encode one-hot
    x = to_categorical(np.squeeze(labels_), num_classes=n_labels)

    # x = np.zeros([h, w, n_labels])
    # for i in range(h):
    #     for j in range(w):
    #         x[i, j, labels_[i][j]] = 1
    # x = x.reshape(h * w, n_labels)
    return x


def decode_onehot(labels_):
    """ input shape (batchsize, h, w, n_labels)
    output shape (batchsize, h, w)
    """
    assert len(labels_.shape) >= 2, "wrong dimension labels, not onehot?"
    return np.argmax(labels_, axis=-1)


# build mapping from labelIds to train mask labels
# build mapping from labelIds to categoryIds
# assume labels ids are sorted ascending
_map_train = []
_map_cat = []
for l in labels:
    # ignore license plate
    if l.id == -1:
        continue
    # training labels are 0-18, use 19 as empty label (255)
    tid = l.trainId
    if tid == 255:
        tid = 19
    _map_train.append(tid)
    # categories are 0-7, use all of them
    cid = l.categoryId
    _map_cat.append(cid)
_map_train = np.array(_map_train)
_map_cat = np.array(_map_cat)

# all training labels get weight 1 and the empty label gets weight 0
train_id_mask_unbal = [1] * 20
train_id_mask = [1] * 19 + [0]

# all categories get weight 1
cat_id_mask = [1] * 8


def label_to_mask(label):
    mask = _map_train[label]
    return mask


def label_to_categories(label):
    mask = _map_cat[label]
    return mask


def labels_to_colors(labels_, n_labels, cat):
    """input labels must already be in the new index sets (training ids or
    categories) and not onehot-encoded

    cat: flag whether category training is enabled in this model or not
    """
    color_mask = make_color_mask(n_labels, cat)
    return apply_color_mask(labels_, color_mask)


def make_color_mask(n_labels, cat):
    # initialize color mask with -1
    color_mask = np.ones((n_labels, 3)) * -1
    for lab in labels:
        # find the id that this labels represents
        new_id = label_to_mask(lab.id)
        if cat:
            new_id = label_to_categories(lab.id)
        # check if color mask is not already filled
        if np.all(color_mask[new_id] == np.array((-1, -1, -1))):
            color_mask[new_id] = lab.color
    assert np.all(color_mask[:] != np.array((-1, -1, -1))), \
        "not all colors found!\n" + str(color_mask)
    return color_mask


def apply_color_mask(img, color_mask):
    """ Input image has shape (h, w, 1) -> not onehot encoded
    color mask: shape (n_labels, 3)
    output: shape (h, w, 3)
    """
    return color_mask[np.squeeze(img)]


if __name__ == '__main__':
    main()
