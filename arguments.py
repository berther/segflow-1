import argparse


def base_parser():
    parser = argparse.ArgumentParser(description="PSPNet LIP dataset")

    # general arguments
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--seed", default=0, type=int,
                        help="random seed. -1 = random seed")
    parser.add_argument("--model_name", default="default_model",
                        type=str, help="modelname")

    # input data configuration and batch size
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1,

                        help="batch size")
    parser.add_argument("--n_classes", type=int, default=20,
                        help="Number of classes (fine labels)")
    parser.add_argument("--n_categories", type=int, default=8,
                        help="Number of categories (coarse labels)")

    # features configuration
    parser.add_argument("--use_flow", action="store_true")
    parser.add_argument("--use_mb", action="store_true")
    parser.add_argument("--use_occ", action="store_true")

    # label configuration
    parser.add_argument("--train_categories", action="store_true",
                        help="train on categories instead of labels")
    parser.add_argument("--unbalanced_weights", action="store_true",
                        help="turn off weight balancing completely")
    parser.add_argument("--unbal_xlast",
                        action="store_true",
                        help="turn off weight balancing except last class")
    parser.add_argument("--zero_weights", action="store_true",
                        help="test zero weights")
    parser.add_argument("--no_norm_images", action="store_true",
                        help="DO NOT normalize images to 0 - 1")

    # images in folders dataset configuration
    parser.add_argument("--data_dir", help="datasets dir", default="datasets")
    parser.add_argument("--dataset", help="dataset name (default city_full)",
                        default="city_full")

    # network architecture stuff
    parser.add_argument("--output_stride", default=16, type=int,
                        help="output stirde")
    parser.add_argument("--output_mode", default="softmax", type=str,
                        help="output activation")
    parser.add_argument("--upsample_type", default="deconv", type=str,
                        help="upsampling type")
    parser.add_argument("--loss", default="categorical_crossentropy",
                        type=str, help="loss function")
    parser.add_argument("--weight_mult", default=1, type=int,
                        help="use bigger loss")
    parser.add_argument("--optimizer", default="adadelta", type=str,
                        help="oprimizer")

    # gpu / cpu hardware configuration
    parser.add_argument("--gpu", default="0", type=str,
                        help="number of gpu")
    parser.add_argument("--use_cpu", action="store_true",
                        help="use cpu instead of gpu")
    # directories
    parser.add_argument("--log_dir", default="logs", type=str,
                        help="log and checkpoint directory")

    # random dataset configuration
    parser.add_argument("--use_random", action="store_true",
                        help="use random data for testing")
    parser.add_argument("--random_val_percent", default=.1, type=float)

    # numpy files dataset configuration
    parser.add_argument("--use_numpy", action="store_true",
                        help="load npy files instead of directories")
    parser.add_argument(
        "--train_mask_npy",
        default="datasets/cityscapes_npy/Train_masks_Cityscapes300.npy")
    parser.add_argument(
        "--train_flow_npy",
        default="datasets/cityscapes_npy/Train_flow_Cityscapes300.npy")
    parser.add_argument(
        "--train_imgs_npy",
        default="datasets/cityscapes_npy/Train_imagesSub_Cityscapes300.npy")

    return parser


def train_parser():
    parser = base_parser()

    parser.add_argument("--n_epochs", default=80, type=int,
                        help="number of epoch")
    parser.add_argument("--epoch_steps", default=3000, type=int,
                        help="number of epoch step")
    parser.add_argument("--val_steps", default=264, type=int,
                        help="number of valdation step")
    parser.add_argument("--restart", action="store_true",
                        help="restart model instead of loading")
    parser.add_argument("--test_generator", action="store_true",
                        help="test data generator")
    return parser


def test_parser():
    parser = base_parser()
    parser.add_argument("--test_dir", default="test_data",
                        type=str,
                        help="directory-path where to tested data is. "
                             "data must include csv files (currently)")
    parser.add_argument("--output_dir", default="result_test",
                        type=str, help="test output path")
    parser.add_argument("--clear_output_dir", action="store_true",
                        help="delete output path first")
    parser.add_argument("--full_size", action="store_true",
                        help="test on full size images")
    parser.add_argument("--full_width", type=int, default=2048,
                        help="width of fullsize images")
    parser.add_argument("--full_height", type=int, default=1024,
                        help="height of fullsize images")

    return parser

def label_to_image_parser():
    parser = base_parser()
    parser.add_argument("--output_dir",default="result_vis")
    parser.add_argument("--cat",action="store_true",
                        help="set this flag to visualize categories")
    return parser