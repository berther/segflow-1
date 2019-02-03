"""File to read the huge cityscapes frames file (zip archive)

filename description:
leftImg8bit_sequence/{split}/{city}/{city}_{imgnum}_{framenum}_leftImg8bit.png

output filenames:
{output_dir}/{split}/{city}_{imgnum}_{framenum}_leftImg8bit.png
"""
import zipfile
import argparse
import os
import shutil


FRAME_MAIN = 19


def main():
    parser = argparse.ArgumentParser("zip file reader")
    parser.add_argument(
        "-f", "--file", type=str, help="input zip file",
        default="datasets/cityscapes/cityscapes_frames.zip")
    parser.add_argument(
        "-od", "--output_dir", type=str, help="output directory",
        default="datasets/cityscapes/extracted_images")
    parser.add_argument(
        "--labels", action="store_true",
        help="process labels (this is step #2 after processing images)")
    parser.add_argument(
        "--singles", action="store_true",
        help="copy images folder and delete second frame")
    parser.add_argument(
        "-n", "--frame_next", type=int, default=20,
        help="which frame to use as second input to optical flow"
             "(default 20 is the next frame after the original image)")
    parser.add_argument(
        "-oi", "--output_dir_images", type=str,
        help="output images subdirectory",
        default="datasets/cityscapes/images")
    parser.add_argument(
        "-os", "--output_dir_images_singles", type=str,
        help="output images subdirectory",
        default="datasets/cityscapes/images_singles")
    parser.add_argument(
        "-il", "--input_dir_labels", type=str,
        help="input labels subdirectory",
        default="datasets/cityscapes/labels_input")
    parser.add_argument(
        "-ol", "--output_dir_labels", type=str,
        help="output labels subdirectory",
        default="datasets/cityscapes/labels_ids")
    parser.add_argument(
        "-oc", "--output_dir_labels_color", type=str,
        help="output colored labels subdirectory",
        default="datasets/cityscapes/labels_color")
    parser.add_argument("-t", "--test", action="store_true")
    args = parser.parse_args()

    # directory where images will be stored
    output_dir = args.output_dir + "/" + args.output_dir_images
    if args.labels:
        # assuming images where already extracted, find corresponding labels
        label_input_dir = args.output_dir + "/" + args.input_dir_labels
        label_output_dir = args.output_dir + "/" + args.output_dir_labels
        label_color_output_dir = \
            args.output_dir + "/" + args.output_dir_labels_color
        find_labels(label_input_dir, label_output_dir, label_color_output_dir,
                    output_dir, test=args.test)
    elif args.singles:
        if args.test:
            raise ValueError("test mode not possible for --singles")
        # copy image folder and delete the second image everywhere
        images_single_output_dir = \
            args.output_dir + "/" + args.output_dir_images_singles
        if not os.path.isdir(images_single_output_dir):
            print("copying image folder (might take a while)...")
            shutil.copytree(output_dir, images_single_output_dir)
        print("purging all images except frame", FRAME_MAIN)
        for root, dirs, files in os.walk(images_single_output_dir):
            for f in files:
                frame = int(f.split("_")[2])
                if frame == FRAME_MAIN:
                    continue
                file_full = root + "/" + f
                os.remove(file_full)
    else:
        # extract the images
        extract_images_from_zip(args.file, output_dir, args.frame_next,
                                test=args.test)


def find_labels(label_input_dir, label_output_dir, label_color_output_dir,
                images_output_dir, test=False):
    print("finding labels...")
    # test has no labels anyway (they are all 0)
    relevant_dirs = ["train", "val"]
    for rd in relevant_dirs:
        curr_dir = images_output_dir + "/" + rd
        # read existing extracted images from the video
        files = os.listdir(curr_dir)
        for i, f in enumerate(files):
            # show progress
            if i % 1000 == 0:
                print("{}: {}/{} ({:.1f}%)".format(
                    rd, i, len(files), i / len(files) * 100))
            # skip everything except the main frame
            frame = int(f.split("_")[2])
            if not frame == FRAME_MAIN:
                continue
            # find labels. the image file looks like this:
            # aachen_000013_000019_leftImg8bit.png
            city, imgnum, framestr = f.split("_")[0:3]
            label_dir = "/".join([label_input_dir, rd, city])
            label_file = "_".join(
                [city, imgnum, framestr, "gtFine_labelIds.png"])
            label_color_file = "_".join(
                [city, imgnum, framestr, "gtFine_color.png"])
            # copy instance id labels
            copy_label(label_dir, label_file, label_output_dir, rd, f,
                       test=test)
            # copy color labels
            copy_label(label_dir, label_color_file, label_color_output_dir, rd,
                       f, test=test)


def copy_label(label_dir, label_file, output_dir, split, image_file,
               test=False):
    target_dir = output_dir + "/" + split
    os.makedirs(target_dir, exist_ok=True)
    source_file = label_dir + "/" + label_file
    target_file = target_dir + "/" + label_file

    if not os.path.exists(source_file):
        print("label not found", source_file)
        print("corresponding image was", image_file)
        raise ValueError("see prints")
    if not test:
        shutil.copy(source_file, target_file)


def extract_images_from_zip(file, output_dir, frame_next, test=False):
    # read zip file
    if not zipfile.is_zipfile(file):
        raise RuntimeError("file is not zip or doesn't exist: " + file)

    # load file tree, ignoring directories
    print("loading", file)
    zf = zipfile.ZipFile(file)

    # some cities have only a large video instead of frames -> totally useless
    skip_cities = [
        "bielefeld", "mainz", "bochum", "hamburg", "hanover", "krefeld",
        "monchengladbach", "strasbourg", "frankfurt", "munster"]

    # preprocess file tree
    files = []
    for a in zf.namelist():
        if not a.endswith(".png"):
            # skip directories
            continue
        city = a.split("/")[2]
        if city in skip_cities:
            # skip useless cities
            continue
        files.append(a)
    files = sorted(files)
    print("found", len(files), "files")

    # frames are numbered 0-29 and represent (-19 to +10) where 0 is the image
    # where we have the ground truth mask
    # for optical flow, we need 0 and +1 => 19 and 20
    get_frames = [FRAME_MAIN, frame_next]

    # loop files
    result_files = []
    for i, f in enumerate(files):
        # show progress
        if i % 1000 == 0:
            print("{}/{} ({:.1f}%)".format(
                i, len(files), i / len(files) * 100))
        if (i + 1) % 1000001 == 0:
            break
        # split up filename and create target filename
        subfolder_split = f.split("/")
        split = subfolder_split[1]  # train, val or test
        fname = subfolder_split[-1]  # filename only
        target_dir = output_dir + "/" + split
        os.makedirs(target_dir, exist_ok=True)
        target_file = target_dir + "/" + fname
        # get frame number and skip irrelevant frames
        frame_num = int(fname.split("_")[2])
        process = False
        if frame_num in get_frames:
            result_files.append(target_file)
            process = True
        # check if the amount of files processed fits (only happens if there
        # are really 30 frames and 2 of them are found)
        if i % 30 == 0:
            files_should_be = i / 15
            if files_should_be != len(result_files):
                print("wrong amount of files", len(result_files), "should be",
                      files_should_be)
                break
        # extract file and write to disk
        if not test and process:
            f_bytes = zf.read(f)
            with open(target_file, "wb") as fh:
                fh.write(f_bytes)

    print(len(result_files), "found for output")


if __name__ == '__main__':
    main()
