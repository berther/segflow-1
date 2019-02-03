import os
import shutil
from keras.callbacks import ModelCheckpoint

# before epoch start
# init up_opt


class MyCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        # keep one additional checkpoint - otherwise, if the script is
        # cancelled while saving the epoch, the model is lost
        if os.path.exists(self.filepath):
            old_filepath = "/".join(
                self.filepath.split("/")[:-1] + ["old_checkpoint.hdf5"])
            shutil.copy(self.filepath, old_filepath)
        # write new checkpoint and do other stuff
        super().on_epoch_end(epoch, logs)
        # save epoch number
        log_path = ""
        for part in self.filepath.split("/")[0:-1]:
            log_path += part + "/"
        log_path += "epochs.txt"
        file = open(log_path, "wt")
        file.write(str(epoch + 1))
        file.close()
