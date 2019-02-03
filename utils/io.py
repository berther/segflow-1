import cv2
import numpy as np
import os


def read_image(file_img, gray=False):
    color_scheme = cv2.IMREAD_COLOR
    if gray:
        color_scheme = cv2.IMREAD_GRAYSCALE
    img = cv2.imread(file_img, color_scheme)
    if img is None:
        raise FileNotFoundError("file", file_img, "not found or corrupt")
    if gray:
        img = np.expand_dims(img, -1)
    return img


def write_image(file_img, img):
    img = np.squeeze(img)
    cv2.imwrite(file_img, img)
    if not os.path.exists(file_img):
        raise FileNotFoundError("couldnt cv2.imwrite " + file_img)


def read_flow(name):
    """read .flo file, code copied from netdef_slim.utils.io
    """
    if name.endswith('.pfm') or name.endswith('.PFM'):
        raise ValueError("PFM files not supported")

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape(
        (height, width, 2))

    return flow.astype(np.float32)


def read_float(name):
    """read .float3 file, code copied from netdef_slim.utils.io
    """
    f = open(name, 'rb')

    if (f.readline().decode("utf-8")) != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))
    return data
