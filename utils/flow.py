import cv2
import numpy as np
import argparse
from utils.io import read_image, read_flow
from utils.flow_vis import flow_to_image


def test_decompress():
    f_orig = ("D:\project_psp\\flow_test\_flow_input\cityflow_1\\"
              "aachen_000000_000019_flow.flo")

    # reconstruct flow from compression
    dir_comp = "flow_test/_flow_compr/cityflow_1/"
    f_comp_x = dir_comp + "aachen_000000_000019_flow.flo_x.png"
    f_comp_y = dir_comp + "aachen_000000_000019_flow.flo_y.png"
    flow_x = read_image(f_comp_x, True)
    flow_y = read_image(f_comp_y, True)
    bounds = (-126.32087707519531, 246.98016357421875,
              -45.00247573852539, 115.37020874023438)
    recon = decompress_flow(flow_x, flow_y, bounds)

    # read original flow
    data = read_flow(f_orig)
    print("ORIG bounds X", np.min(data[:, :, 0]),
          np.max(data[:, :, 0]), "Y", np.min(data[:, :, 1]),
          np.max(data[:, :, 1]))

    # compress
    flow_x, flow_y, bounds = compress_flow(data)
    print("COMPRESS bounds X", np.min(flow_x),
          np.max(flow_x), "Y", np.min(flow_y),
          np.max(flow_y))

    # uncompress
    # recon = decompress_flow(flow_x, flow_y, bounds)

    # compare
    for d in range(2):
        print(data[0, 0, d])
        print(recon[0, 0, d])
    print(np.sum(np.abs(data - recon)))

    # make images and show
    orig_bgr = flow_to_image(data)
    recon_bgr = flow_to_image(recon)

    # show them
    compare = np.concatenate((orig_bgr, recon_bgr), axis=1)
    cv2.imshow("orig", compare)
    cv2.waitKey(0)


def compress_flow(flow):
    minx = np.min(flow[:, :, 0])
    maxx = np.max(flow[:, :, 0])
    miny = np.min(flow[:, :, 1])
    maxy = np.max(flow[:, :, 1])
    flow_x = np.uint8((flow[:, :, 0] - minx) / (maxx - minx) * 255)
    flow_x = np.expand_dims(flow_x, -1)
    flow_y = np.uint8((flow[:, :, 1] - miny) / (maxy - miny) * 255)
    flow_y = np.expand_dims(flow_y, -1)
    return flow_x, flow_y, (minx, maxx, miny, maxy)


def decompress_flow(flow_x, flow_y, flow_bounds):
    """
    :param flow_x: grayscale flow x image (0-255), (height, width, 1)
    :param flow_y: grayscale flow y image (0-255), (height, width, 1)
    :param flow_bounds: original flow bounds
                        (f_x_min, f_x_max, f_y_min, f_y_max)
    :return: flow, 2d array with original values from flownet
    """
    f_x_min, f_x_max, f_y_min, f_y_max = flow_bounds

    # restore flow bounds from 0-255 greyscale
    flow_x = (flow_x / 255) * (f_x_max - f_x_min) + f_x_min
    flow_y = (flow_y / 255) * (f_y_max - f_y_min) + f_y_min
    flow = np.concatenate((flow_x, flow_y), axis=2)
    return flow


if __name__ == '__main__':
    test_decompress()
