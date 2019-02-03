import json
import os
import random

import keras.backend.tensorflow_backend as KTF
# import pandas as pd  # only used for images list, npy used here instead
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, CSVLogger
from keras.models import load_model

from arguments import train_parser
from custom_metric import mean_iou_func
from custom_loss import weighted_categorical_crossentropy
from generator import data_gen_random, DataStorageNpy, DataStorage
from model import PSPNet50
from my_checkpoint import MyCheckpoint

from utils.find_input import find_input, find_masks


def main():
    parser = train_parser()
    args = parser.parse_args()

    if args.use_cpu:
        # disable gpu completely
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # set device number
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.seed != -1:
        # set random seed
        np.random.seed(args.seed)
        random.seed(args.seed)
        tf.set_random_seed(args.seed)

    # get old session old_session = KTF.get_session()

    with tf.Graph().as_default():
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)

        # determine some variables from args
        dims, input_shape, n_labels, model_dir, checkpoint_path = find_input(
            args)

        os.makedirs(model_dir, exist_ok=True)
        # set callbacks
        # for multiple checkpoints set filepath to "...point{epoch:d}.hdf5"
        cp_cb = MyCheckpoint(
            filepath=model_dir + "/checkpoint.hdf5",
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
            mode='auto',
            period=1)
        tb_cb = TensorBoard(
            log_dir=model_dir,
            write_images=True)
        log = CSVLogger(
            model_dir + "/training.log",
            append=True
        )

        # set generator
        if args.use_random:
            # create random data
            train_gen = data_gen_random(dims, n_labels, args.batch_size,
                                        args.use_flow)
            val_gen = data_gen_random(dims, n_labels, args.batch_size,
                                      args.use_flow)
        elif args.use_numpy:
            # load small numpy dataset
            n_labels = args.n_classes
            datagen = DataStorageNpy(
                args.train_imgs_npy, args.train_mask_npy, args.train_flow_npy,
                args.val_percent, dims, n_labels, args.use_flow,
                args.batch_size)

            train_gen = datagen.generate(training=True)
            val_gen = datagen.generate(training=False)
        else:
            # load dataset from folders
            datagen = DataStorage(
                args.data_dir, args.dataset, dims, n_labels)
            train_gen = datagen.generate(
                args.batch_size, args.use_flow, args.use_mb, args.use_occ,
                args.train_categories, split="train", shuffle=True,
                normalize_images=not args.no_norm_images)
            val_gen = datagen.generate(
                args.batch_size, args.use_flow, args.use_mb, args.use_occ,
                args.train_categories, split="val", shuffle=False,
                random_crop=False,
                normalize_images=not args.no_norm_images)
            if args.test_generator:
                imgs, labels = next(train_gen)
                print("datagen yielded", imgs.shape, labels.shape)
                return
            # # test has flow and images, but masks are zero
            # # so it cannot be used to compute accuracy
            # test_gen = datagen.generate(args.batch_size, args.use_flow,
            #                             args.use_mb, args.use_occ,
            #                             split="test")

        # ----- determine class weights
        weight_mask, weight_mask_iou = find_masks(args, n_labels)
        print("using weight mask:", weight_mask)
        print("using weight mask for iou:", weight_mask_iou)

        # ----- define loss
        old_loss = args.loss
        print("old loss", old_loss)
        loss = weighted_categorical_crossentropy(weight_mask, args.weight_mult)

        # ----- define iou metric
        def mean_iou(y_true, y_pred):
            return mean_iou_func(
                y_true, y_pred, args.batch_size, dims,
                n_labels, weight_mask_iou, iou_test=False)

        # ----- restart or load model
        restart = args.restart
        if not args.restart:
            # check if model is available
            if not os.path.exists(checkpoint_path):
                print("no previous model available, restarting from epoch 0")
                restart = True

        print("model input shape", input_shape)
        if restart:
            # set model
            pspnet = PSPNet50(
                input_shape=input_shape,
                n_labels=n_labels,
                output_mode=args.output_mode,
                upsample_type=args.upsample_type)

            # compile model
            pspnet.compile(
                loss=loss,
                optimizer=args.optimizer,
                metrics=["accuracy", mean_iou])
            # metrics=["acc_iou", mean_iou])
            starting_epoch = 0

        else:
            # load model from dir
            try:
                pspnet = load_model(checkpoint_path, custom_objects={
                    'mean_iou': mean_iou,
                    'loss': loss})
            except OSError:
                raise OSError("failed loading checkpoint", checkpoint_path)
            except ValueError:
                raise ValueError(
                    "possible the checkpoint file is corrupt because the "
                    "script died while saving. use the backup old_checkpoint")
            try:
                with open(model_dir + "/epochs.txt", "r") as fh:
                    starting_epoch = int(fh.read())
            except OSError:
                raise FileNotFoundError("could not find epochs.txt")
            print("reloading model epoch", starting_epoch)

        # print model summary (weights, layers etc)
        if args.summary:
            print(pspnet.summary())

        # ----- fit with genarater
        pspnet.fit_generator(
            generator=train_gen,
            steps_per_epoch=args.epoch_steps,
            epochs=args.n_epochs,
            validation_data=val_gen,
            validation_steps=args.val_steps,
            callbacks=[cp_cb, tb_cb, log],
            initial_epoch=starting_epoch
        )


if __name__ == "__main__":
    main()
