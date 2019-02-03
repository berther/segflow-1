import os
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser(description="PSPNet LIP dataset")
    parser.add_argument("--log_dir", default="logs", type=str,
                        help="log and checkpoint directory")
    parser.add_argument("--target_log_dir", default="logs_train", type=str,
                        help="output train log directory")
    args = parser.parse_args()

    log_dir = args.log_dir
    target_log_dir = args.target_log_dir
    os.makedirs(target_log_dir, exist_ok=True)

    for model_name in os.listdir(log_dir):
        full_dir = log_dir + "/" + model_name
        train_log = full_dir + "/training.log"
        out_log = target_log_dir + "/" + model_name + ".log"
        if not os.path.exists(train_log):
            print(train_log,
                  "not found.\npossibly this model was not trained.")
            continue
        content = open(train_log, "rt").read()
        if len(content) == 0:
            print(train_log, "empty.\npossibly this model was not trained.")
            continue
        shutil.copy(train_log, out_log)
        print("created", out_log)


if __name__ == '__main__':
    main()
