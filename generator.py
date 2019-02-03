import csv
import os
import random
import string

import numpy as np

from utils.flow import flow_to_image
from utils.imgs import center_crop_batch, crop, rescale, rescale_center_crop
from utils.io import read_image, write_image
from utils.labels import encode_onehot, label_to_categories, label_to_mask, \
    labels_to_colors


def write_network_input(
        write_folder, write_file_base, img, flow_x, flow_y, motion_bd,
        occlusion, mask, use_flow, use_mb, use_occ, n_labels, train_categories,
        write_all=True):
    """write network input (before or after cropping) to files"""
    os.makedirs(write_folder, exist_ok=True)
    # make a counter to only write the first 5 or so images
    file_base = write_folder + "/" + write_file_base
    write_image(file_base + "img.png", img)
    # default: only write the real input that is used instead of everything
    # with write_all=True, always writes everything
    if use_flow or write_all:
        write_image(file_base + "flow.png", flow_to_image(
            np.concatenate((flow_x, flow_y), -1)))
    if use_mb or write_all:
        write_image(file_base + "motion_bd.png", motion_bd)
    if use_occ or write_all:
        write_image(file_base + "occlusion.png", occlusion)
    # visualize and write mask
    write_image(file_base + "segmask.png", labels_to_colors(
        mask, n_labels, train_categories))


class DataStorage(object):
    def __init__(self, data_dir, dataset, dims, n_labels):
        self.data_dir = data_dir
        self.dataset = dataset
        self.dataset_dir = self.data_dir + "/" + self.dataset
        self.imgs_dir = self.dataset_dir + "/" + "images"
        self.labels_dir = self.dataset_dir + "/" + "labels_ids"
        self.flow_dir = self.dataset_dir + "/" + "flow"
        self.dims = dims
        self.n_labels = n_labels

    def generate(self, batch_size, use_flow, use_mb, use_occ,
                 train_categories, split="train",
                 shuffle=True, random_crop=True,
                 write_folder="", write_max=5, write_cropped=True,
                 write_uncropped=False, normalize_images=False,
                 full_size=False):
        """ data generator.
        set write_max = -1 to write all input images back to disk
        """
        # load csv file as numpy string array [n_datapoints, m_features]
        csv_file = self.dataset_dir + "/" + split + ".csv"
        csv_content = []
        with open(csv_file, 'rt') as fh:
            csv_reader = csv.reader(fh, delimiter=";", quotechar="|")
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue
                csv_content.append(row)
        csv_content = np.array(csv_content)
        len_data = len(csv_content)
        current_idx = np.arange(len_data)
        if shuffle:
            np.random.shuffle(current_idx)
        counter = 0
        counter_write_images = 0
        while True:
            imgs, labels = [], []
            channels = 3 + use_flow * 2 + use_mb + use_occ
            for i in range(batch_size):
                row = csv_content[current_idx[counter]]
                file_base = "_".join(row[:3]) + "_"

                # ----- load image
                file_img = "/".join(
                    [self.imgs_dir, split, file_base + "leftImg8bit.png"])
                img = read_image(file_img)
                oh, ow, img_channels = img.shape
                h, w = self.dims

                # ----- load flow
                file_flow_base = "/".join([self.flow_dir, split, file_base])
                flow_x = read_image(file_flow_base + "flow.flo_x.png", True)
                flow_y = read_image(file_flow_base + "flow.flo_y.png", True)
                # load original flow value bounds from csv file
                f_x_min, f_x_max, f_y_min, f_y_max = \
                    (float(a) for a in row[3:])
                # restore flow bounds from 0-255 greyscale
                flow_x = (flow_x / 255) * (f_x_max - f_x_min) + f_x_min
                flow_y = (flow_y / 255) * (f_y_max - f_y_min) + f_y_min
                # normalize over original image width so that
                # -0.5 in flow_x means the pixel flows half image width to left
                flow_x /= ow
                flow_y /= oh

                # ----- load motion boundaries, occlusion
                motion_bd = read_image(file_flow_base + "mb_soft.png", True)
                occlusion = read_image(file_flow_base + "occ_soft.png", True)

                # ----- load labels
                label = read_image("/".join([
                    self.labels_dir, split,
                    file_base + "gtFine_labelIds.png"]), gray=True)
                # transform to classes or categories
                mask = label_to_mask(label)
                if train_categories:
                    mask = label_to_categories(label)

                if write_folder != "" and write_uncropped and (
                        counter_write_images < write_max or write_max == -1):
                    # write uncropped images to some test folder
                    write_file_base = file_base + "orig_"
                    write_network_input(
                        write_folder, write_file_base, img, flow_x, flow_y,
                        motion_bd, occlusion, mask, use_flow, use_mb, use_occ,
                        self.n_labels, train_categories)

                if full_size:
                    # for full size images set target size to original size
                    # and dont crop
                    h, w = oh, ow
                elif random_crop:
                    # ----- for training, random rescale and random crop
                    # define a random rescale 0.5 - 2
                    scale_log2 = np.random.uniform(-1, 1)
                    scale = 2 ** scale_log2

                    # set cropsize so that scaling that crop back to the target
                    # input width yields the desired scaling
                    hs = np.round(h / scale).astype(int)
                    ws = np.round(w / scale).astype(int)

                    # define random crop (+1 because randint is high exclusive)
                    try:
                        posy = np.random.randint(oh - hs + 1)
                        posx = np.random.randint(ow - ws + 1)
                    except ValueError:
                        raise ValueError(
                            "random crop with negative or zero bounds "
                            "{}x{}, {}x{}".format(oh, ow, hs, ws))

                    # crop and rescale everything
                    img = crop(img, posy, posx, hs, ws)
                    img = rescale(img, h, w)
                    flow_x = crop(flow_x, posy, posx, hs, ws)
                    flow_x = rescale(flow_x, h, w, gray=True)
                    flow_y = crop(flow_y, posy, posx, hs, ws)
                    flow_y = rescale(flow_y, h, w, gray=True)
                    motion_bd = crop(motion_bd, posy, posx, hs, ws)
                    motion_bd = rescale(motion_bd, h, w, gray=True)
                    occlusion = crop(occlusion, posy, posx, hs, ws)
                    occlusion = rescale(occlusion, h, w, gray=True)
                    mask = crop(mask, posy, posx, hs, ws)
                    mask = rescale(mask, h, w, gray=True, nearest=True)
                else:
                    # ----- for validation and test, center crop and downscale
                    img = rescale_center_crop(img, h, w)
                    flow_x = rescale_center_crop(flow_x, h, w, gray=True)
                    flow_y = rescale_center_crop(flow_y, h, w, gray=True)
                    motion_bd = rescale_center_crop(motion_bd, h, w, gray=True)
                    occlusion = rescale_center_crop(occlusion, h, w, gray=True)
                    mask = rescale_center_crop(mask, h, w, gray=True,
                                               nearest=True)

                if write_folder != "" and write_cropped and (
                        counter_write_images < write_max or write_max == -1) \
                        and not full_size:
                    # write cropped images to some test folder
                    write_file_base = file_base + "crop_"
                    write_network_input(
                        write_folder, write_file_base, img, flow_x, flow_y,
                        motion_bd, occlusion, mask, use_flow, use_mb, use_occ,
                        self.n_labels, train_categories)

                # ----- finalize image
                # stack image and features to one big input
                if normalize_images:
                    img = img.astype(np.float32) / 255.0

                vstack = [img]
                if use_flow:
                    vstack += [flow_x, flow_y]
                if use_mb:
                    vstack.append(motion_bd)
                if use_occ:
                    vstack.append(occlusion)
                img = np.concatenate(vstack, axis=2)
                # assert correct final input shape
                assert img.shape == (h, w, channels)
                # collect image into batch
                imgs.append(img)

                # ----- finalize mask
                # assert in label to mask the correct size
                # currently it does implicit cropping which is bad
                mask = encode_onehot(mask, h, w, n_labels=self.n_labels)
                assert mask.shape == (h * w, self.n_labels)
                labels.append(mask)

                # check end of data reached
                counter += 1
                if counter == len_data:
                    if shuffle:
                        # shuffle dataset
                        np.random.shuffle(current_idx)
                    # start again
                    counter = 0
                counter_write_images += 1
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels


class DataStorageNpy(object):
    def __init__(self, train_imgs_npy, train_mask_npy, train_flow_npy,
                 val_percent, dims, n_labels, use_flow, batch_size):
        """Load npy data and randomly split into train/val set"""
        print("loading npy data: images")
        imgs = np.load(train_imgs_npy)
        imgs = center_crop_batch(imgs, *dims)
        len_data = len(imgs)
        idx_order = np.random.permutation(len_data)
        split_at = int((1 - val_percent) * len_data)
        train_idx = idx_order[:split_at]
        val_idx = idx_order[split_at:]
        self.len_data_train = len(train_idx)
        self.len_data_val = len(val_idx)
        print("data: {} train, {} val".format(
            self.len_data_train, self.len_data_val))
        self.train_imgs = np.copy(imgs[train_idx])
        self.val_imgs = np.copy(imgs[val_idx])
        del imgs

        print("loading masks")
        mask = np.load(train_mask_npy)
        mask = center_crop_batch(mask, *dims)
        self.train_mask = np.copy(mask[train_idx])
        self.val_mask = np.copy(mask[val_idx])
        del mask

        if use_flow:
            print("loading flow")
            flow = np.load(train_flow_npy)
            flow = center_crop_batch(flow, *dims)
            self.train_flow = np.copy(flow[train_idx])
            self.val_flow = np.copy(flow[val_idx])
            del flow
        else:
            self.train_flow = None
            self.val_flow = None

        self.dims = dims
        self.n_labels = n_labels
        self.use_flow = use_flow
        self.len_data = len_data
        self.batch_size = batch_size

    def generate(self, training=True):
        if training:
            imgs, mask, flow, len_data = (
                self.train_imgs, self.train_mask, self.train_flow,
                self.len_data_train)
        else:
            imgs, mask, flow, len_data = (
                self.val_imgs, self.val_mask, self.val_flow, self.len_data_val)
        counter = 0
        while True:
            imgs_arr, mask_arr = [], []
            for i in range(self.batch_size):
                # add the next data pair to the lists
                img = imgs[counter]
                if self.use_flow:
                    flow_img = flow[counter]
                    img = np.concatenate((img, flow_img), axis=2)
                imgs_arr.append(img)
                mask0 = mask[counter]
                # print("mask0 input",mask0.shape)
                array_mask = encode_onehot(mask0, *self.dims, self.n_labels)
                # print("array mask output",array_mask.shape)
                mask_arr.append(array_mask)
                counter += 1

                if counter == len_data:
                    # end of data reached: shuffle and start again
                    np.random.shuffle(imgs)
                    np.random.shuffle(mask)
                    if self.use_flow:
                        np.random.shuffle(flow)
                    counter = 0
            imgs_arr = np.array(imgs_arr)
            mask_arr = np.array(mask_arr)
            # print("datagen yielded", imgs_arr.shape, mask_arr.shape)
            yield imgs_arr, mask_arr


def data_gen_random(dims, n_labels, batch_size, use_flow):
    """Random data generator for testing if the model runs"""
    while True:
        imgs, labels = [], []
        channels = 3
        if use_flow:
            channels = 5
        img_size = (dims[0], dims[1], channels)
        mask_size = (dims[0], dims[1], 1)
        for i in range(batch_size):
            rand_img = np.random.rand(*img_size)
            imgs.append(rand_img)

            rand_labels = np.random.randint(0, n_labels, mask_size)
            rand_mask = encode_onehot(rand_labels, *dims, n_labels)
            labels.append(rand_mask)
        imgs = np.array(imgs)
        labels = np.array(labels)
        # print("random datagen yielded",imgs.shape, labels.shape)
        yield imgs, labels
