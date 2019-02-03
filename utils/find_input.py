import json
from utils.labels import train_id_mask, train_id_mask_unbal, cat_id_mask


def find_input(args):
    # find input shape
    dims = [args.img_height, args.img_width]
    channels = 3
    if args.use_flow:
        channels += 2
    if args.use_mb:
        channels += 1
    if args.use_occ:
        channels += 1
    input_shape = (*dims, channels)

    # find n_labels
    n_labels = args.n_classes
    if args.train_categories:
        n_labels = args.n_categories

    # find model directory
    model_dir = args.log_dir + "/" + args.model_name
    checkpoint_path = model_dir + "/checkpoint.hdf5"

    return dims, input_shape, n_labels, model_dir, checkpoint_path


def find_masks(args, n_labels):
    if args.use_random or args.use_numpy:
        # for these, do not need weights
        weight_mask = [1] * n_labels
        weight_mask_iou = [1] * (n_labels - 1) + [0]
    else:
        # the actual cityscapes dataset needs a weight mask
        if args.unbal_xlast:
            if args.train_categories:
                weight_mask = cat_id_mask
                weight_mask_iou = cat_id_mask
            else:
                weight_mask = train_id_mask
                weight_mask_iou = train_id_mask
        elif args.unbalanced_weights:
            if args.train_categories:
                weight_mask = cat_id_mask
                weight_mask_iou = cat_id_mask
            else:
                weight_mask = train_id_mask_unbal
                weight_mask_iou = train_id_mask_unbal
        else:
            # load precomputed balanced weights from json files
            base_dir = args.data_dir + "/" + args.dataset
            if args.train_categories:
                json_file = base_dir + "/weights_categories.json"
                weight_mask_iou = cat_id_mask
            else:
                json_file = base_dir + "/weights_labels.json"
                weight_mask_iou = train_id_mask
            with open(json_file, "rt") as fh:
                weight_mask = json.load(fh)
    if args.zero_weights:
        weight_mask = [0] * n_labels
    return weight_mask, weight_mask_iou
