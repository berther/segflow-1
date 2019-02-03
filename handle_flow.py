import argparse
import os
import shutil
import numpy as np
import cv2


FRAME_MAIN = 19


def main():
    parser = argparse.ArgumentParser("cityscapes flow computation")
    parser.add_argument("--image_dir", type=str, default="images_for_flow",
                        help="cityscapes frames input")
    parser.add_argument("--flow_dir", type=str, default="result_flow",
                        help="where to put the flow")
    parser.add_argument("--compress", action="store_true",
                        help="compress existing flow (this is step 2 after"
                             "creating the raw flow data)")
    parser.add_argument("--compress_dir", type=str,
                        default="result_flow_compressed",
                        help="where to put the compressed flow")
    parser.add_argument("--next_frame", type=int, default=20,
                        help="next frame after original frame")
    parser.add_argument("-t", "--test", action="store_true")

    args = parser.parse_args()
    if args.compress:
        compress_all_flow(args.flow_dir, args.compress_dir, test=args.test)
    else:
        flow_folder(args.image_dir, args.flow_dir, args.next_frame,
                    test=args.test)


def compress_all_flow(flow_input_dir, compressed_output_dir, test=False):
    bounds_max = np.zeros(4)

    from utils.io import read_flow, read_float
    from utils.flow import compress_flow

    for root, dirs, files in os.walk(flow_input_dir):
        print("walking at", root)
        root = root.replace("\\", "/")
        split = root.split("/")[-1]
        files = sorted(files)
        if split == "":
            # skip main dir
            continue
        out_dir = "/".join([compressed_output_dir, split])
        os.makedirs(out_dir, exist_ok=True)
        # train.csv: city, imgnumstr, framestr, flow_x_min,  flow_x_max, ...
        csv_file = compressed_output_dir + "/" + split + ".csv"
        print("csv file", csv_file)
        fh = open(csv_file, "wt")
        fh.write("city;image_number;frame;flow_x_min;flow_x_max;"
                 "flow_y_min;flow_y_max\n")
        if not test:
            os.makedirs(out_dir, exist_ok=True)
        for f in files:
            file_full = root + "/" + f
            if f.endswith(".flo"):
                # read optical flow
                print("found flow file", file_full, f)
                data = read_flow(file_full)
                flow_x, flow_y, bounds = compress_flow(data)
                city, img_num, frame, _ = f.split("_")
                # compute maximum flow over all images
                bounds_max = np.maximum(bounds_max, np.abs(np.array(bounds)))
                # write to csv file
                fh.write("{};{};{};{};{};{};{}\n".format(
                    city, img_num, frame, *bounds))
                # save compressed flow
                if not test:
                    file_x = f + "_x.png"
                    file_y = f + "_y.png"
                    cv2.imwrite(out_dir + "/" + file_x, flow_x)
                    cv2.imwrite(out_dir + "/" + file_y, flow_y)
            elif f.endswith(".float3"):
                # these are all already scaled to 0-1
                data = read_float(file_full) * 255
                if not test:
                    compr_file = f.replace(".float3", ".png")
                    cv2.imwrite(out_dir + "/" + compr_file, data)
            else:
                print("ignoring file", file_full)

        fh.close()
    # output bounds
    for i in range(2):
        print("MAX bounds for dim {}: -{:.3f} to {:.3f}".format(
            i, bounds_max[i * 2], bounds_max[i * 2 + 1]))


def flow_folder(image_dir, flow_output_dir, next_frame, test=False):
    """run FLowNet3 on the image pairs through the system pipe
    restart everything for each image pair

    loading the controller as described in netdef_models didnt work
    """
    fn_dict = {
        'flow[0].fwd': ("flow.flo", 'flow[0].fwd.flo'),
        'occ[0].fwd': ("occ_hard.float3", 'occ[0].fwd.float3'),
        'occ_soft[0].fwd': ("occ_soft.float3", 'occ_soft[0].fwd.float3'),
        'mb[0].fwd': ("mb_hard.float3", 'mb[0].fwd.float3'),
        'mb_soft[0].fwd': ("mb_soft.float3", 'mb_soft[0].fwd.float3')}
    tmp_dir = "tmp_flow"
    if not test:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            root = root.replace("\\", "/")
            city, imgnum, framestr, rest = f.split("_")
            frame = int(framestr)
            if not frame == FRAME_MAIN:
                continue
            img0 = root + "/" + "_".join([city, imgnum, framestr, rest])
            img1 = root + "/" + "_".join(
                [city, imgnum, "{:06d}".format(next_frame), rest])
            print("calculating flow...")
            print(img0)
            print(img1)
            if not test:
                # out = c.net_actions.eval(img0, img1)
                os.system("python controller.py eval {} {} {}".format(
                    img0, img1, tmp_dir
                ))

            target_dir = flow_output_dir + "/" + root.split("/")[-1]
            os.makedirs(target_dir, exist_ok=True)

            for key, val in fn_dict.items():
                rest = val[0]
                source_file = val[1]
                source_file_full = tmp_dir + "/" + source_file
                target_file_full = target_dir + "/" + "_".join(
                    [city, imgnum, framestr, rest])
                print("moving", source_file_full, "to", target_file_full)
                shutil.move(source_file_full, target_file_full)


if __name__ == '__main__':
    main()
