"""
Problem: DataGen braucht im Moment ein csv-file sonst funktioniert er nicht
Sobald wir Flow haben brauchen wir das CSV für flow decompression

Fuer models mit NUR image könnte man ohne CSV predicten:
- mit nem flag args.predict_image anstossen:
    - statt datagenerator mit args.data_dir und args.dataset
    über das args.test_input folder loopen
    - image und mask laden und center croppen (code aus datagen kopieren dafür)
    - predicten

Profilösung wäre, das flownet erst anwenden und dann unser model, das ist aber
zu viel arbeit hier

"""
import os
import shutil
import time
import numpy as np
import csv
import tensorflow as tf
from keras.models import load_model
import keras.backend.tensorflow_backend as KTF
from arguments import test_parser
from custom_metric import mean_iou_func
from generator import DataStorage
from utils.find_input import find_input, find_masks
from custom_loss import weighted_categorical_crossentropy
from utils.io import write_image
from utils.labels import label_to_categories, label_to_mask, make_color_mask, \
    apply_color_mask, decode_onehot

from model import PSPNet50
from iou_test.iou import compute_mean_iou


VAL_STEPS = 264


def main():
    """
    Usage: python test.py --model_name $Modelname$  --train_categories
    gets val data and predicts with model $Modelname$
    stores rgb sgmentations in logs/$Modelname$/pred
    """
    parser = test_parser()
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("Evaluation is not defined for "
                         "batch size other than 1.")

    # determine some variables from args
    dims, input_shape, n_labels, model_dir, checkpoint_path = find_input(
        args)
    weight_mask, weight_mask_iou = find_masks(args, n_labels)

    # determine output directory
    output_dir = args.output_dir + "/" + args.model_name
    if args.clear_output_dir:
        shutil.rmtree(output_dir, ignore_errors=True)
        time.sleep(1)
    os.makedirs(output_dir, exist_ok=True)

    max_vis = 264
    max_calc = VAL_STEPS

    # load data gen
    datagen = DataStorage(
        args.data_dir, args.dataset, dims, n_labels)
    train_gen = datagen.generate(
        args.batch_size, args.use_flow, args.use_mb, args.use_occ,
        args.train_categories, split="train", shuffle=True,
        normalize_images=not args.no_norm_images)
    val_gen = datagen.generate(
        args.batch_size, args.use_flow, args.use_mb, args.use_occ,
        args.train_categories, split="val", shuffle=False, random_crop=False,
        write_folder=output_dir, write_cropped=True, write_uncropped=True,
        write_max=max_vis, normalize_images=not args.no_norm_images,
        full_size=args.full_size)

    # setup model
    loss = weighted_categorical_crossentropy(weight_mask, args.weight_mult)

    def mean_iou(y_true, y_pred):
        return mean_iou_func(
            y_true, y_pred, args.batch_size, dims,
            n_labels, weight_mask_iou, iou_test=False)

    # load and print model epoch
    try:
        model_epoch = int(open(model_dir + "/epochs.txt").read())
    except OSError:
        raise FileNotFoundError("could not find epochs.txt")
    print("reloading model epoch", model_epoch)

    if args.full_size:
        # for full size models, load weights only instead of full model
        # check weight age
        weight_info = model_dir + "/epochs_weight.txt"
        weight_path = model_dir + "/weights.h5"
        if not os.path.exists(weight_info):
            weight_epoch = -1
        else:
            weight_epoch = int(open(weight_info, "rt").read())
        if weight_epoch != model_epoch:
            print("weight epoch is", weight_epoch, "(-1 = no weights)")
            print("load model and save weights...")
            try:
                pspnet = load_model(checkpoint_path, custom_objects={
                    'mean_iou': mean_iou,
                    'loss': loss})
            except OSError:
                print(checkpoint_path + " does not exist")
                return
            print("model loaded, saving weights...")
            pspnet.save_weights(weight_path, overwrite=True)
            open(weight_info, "wt").write(str(model_epoch))
            # try to free up all that memory that the model consumed
            del pspnet
            KTF.clear_session()
        # now weights should be correct. load them
        channels = 3
        if args.use_flow:
            channels += 2
        if args.use_mb:
            channels += 1
        if args.use_occ:
            channels += 1
        input_shape = (args.full_height, args.full_width, channels)
        pspnet = PSPNet50(
            input_shape=input_shape,
            n_labels=n_labels,
            output_mode=args.output_mode,
            upsample_type=args.upsample_type)
        pspnet.compile(
            loss=loss,
            optimizer=args.optimizer,
            metrics=["accuracy", mean_iou])
        pspnet.load_weights(weight_path)
        print("loaded weights and created full size model")

    else:
        # regular size models
        try:
            pspnet = load_model(checkpoint_path, custom_objects={
                'mean_iou': mean_iou,
                'loss': loss})
        except OSError:
            print(checkpoint_path + " does not exist")
            return

    # check weights for nans
    for i, l in enumerate(pspnet.layers):
        ws = l.get_weights()
        for j, w in enumerate(ws):
            # print("weights shaped", w.shape)
            nans = np.isnan(w)
            nans_c = np.sum(nans)
            if nans_c > 0:
                print("layer", i)
                print(l)
                print("weights", j, "nans", nans_c)

    os.makedirs(model_dir + '/pred', exist_ok=True)
    color_mask = make_color_mask(n_labels, args.train_categories)

    print("saving first", max_vis, "inputs to",
          output_dir + "/" + args.model_name)
    print("categories?", args.train_categories)

    evalus = []

    verbose = 1
    preds, targets = [], []
    for i in range(VAL_STEPS):

        # print progress
        if i % 10 == 0:
            print("{}/{} ({:.1f}%)".format(i, VAL_STEPS, i / VAL_STEPS * 100))

        # get data
        x, target = next(val_gen)
        print("x", x.shape, "y", target.shape)
        continue

        # predict and save evaluation (loss etc.)
        prediction = pspnet.predict(x, batch_size=1, verbose=verbose)
        evalu = pspnet.evaluate(x, target, batch_size=args.batch_size)
        evalus.append(evalu)

        # only calculate on the first Y images then break
        if i >= max_calc:
            break

        # only visualize the first X images
        if i >= max_vis:
            verbose = 0
            continue

        print("PRED", prediction)

        # reshape prediction and target
        if args.full_size:
            h = args.full_height
            w = args.full_width
        else:
            h = args.img_height
            w = args.img_width

        prediction = np.reshape(prediction, (
            args.batch_size, h, w, n_labels))
        target = np.reshape(target, (
            args.batch_size, h, w, n_labels))

        # color prediction and target
        img = decode_onehot(prediction)
        mask = decode_onehot(target)
        preds.append(img)
        targets.append(mask)
        img_new = apply_color_mask(img, color_mask)
        mask_new = apply_color_mask(mask, color_mask)

        # save prediction and target
        full_str = ""
        if args.full_size:
            full_str = "full"
        img_file = model_dir + '/pred/pred' + full_str + str(i) + '.png'
        mask_file = model_dir + '/pred/pred' + full_str + str(i) + 'true.png'
        write_image(img_file, img_new)
        write_image(mask_file, mask_new)
        print("saved prediction", i, "to", img_file)

    # evaluate metrics
    names = ["val loss", "val acc", "val mean iou"]
    evalus = np.array(evalus)
    evalus_mean = np.mean(evalus, axis=1)
    print()
    print("averages over each individual prediction (wrong values)")
    for i, n in enumerate(names):
        print("{:20} {:.4f}".format(n, evalus_mean[i]))

    # manually compute iou
    targets = np.array(targets)
    preds = np.array(preds)
    print()
    print("mean numpy iou over everything with weights")
    iou, seen_c = compute_mean_iou(
        targets, preds, n_labels, weights=weight_mask_iou, return_seen=True)

    print("{:20} {:.4f} seen {:2d} classes".format(
        "numpy val mean iou", iou, seen_c))

    print()
    print("class predictions numpy iou")
    ious = []
    for i in range(n_labels):
        test_weight_mask = np.zeros(n_labels)
        test_weight_mask[i] = 1
        iou, seen_c = compute_mean_iou(
            targets, preds, n_labels, weights=test_weight_mask,
            return_seen=True, divide_over_num_weights=True)
        print("iou class {:3d}: {:.4f} seen {:2.0f} classes".format(
            i, iou, seen_c))
        ious.append(iou)
    print("final iou {:4f}".format(np.mean(ious)))

    print()
    csv_file = model_dir + "/training.log"
    print("last entry from training log")
    fh = open(csv_file, "rt")
    csv_reader = csv.reader(fh)
    title_row = next(csv_reader)
    row = None
    for row in csv_reader:
        pass
    for title, val in zip(title_row, row):
        print("{:20} {:.4f}".format(title, float(val)))

    print()
    print("test it with evaluate_generator")
    evalus2 = pspnet.evaluate_generator(val_gen, VAL_STEPS, max_queue_size=1)
    for i, n in enumerate(names):
        print("{:20} {:.4f}".format(n, evalus2[i]))

    print("done testing.")


if __name__ == "__main__":
    main()
