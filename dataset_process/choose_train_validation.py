# 根据比例将all.csv的所有标签分成train.csv和val.csv, 并打乱顺序
import os
import random


def main():
    root_path = r"F:\My_Resources\datasets\GestureRecognition_fwwb\jester"
    train_percent = 0.81
    val_percent = 1 - train_percent

    all_file_p = os.path.join(root_path, "all.csv")
    train_file_p = os.path.join(root_path, "train.csv")
    val_file_p = os.path.join(root_path, "val.csv")

    all_lines = []
    with open(all_file_p, "r") as all_file:
        for line in all_file:
            all_lines.append(line.replace("\n", ""))
    random.shuffle(all_lines)
    with open(train_file_p, "w") as train_file, open(val_file_p, "w") as val_file:
        for sign in all_lines:
            if random.random() < train_percent:
                train_file.write(sign + "\n")
                train_file.flush()
            else:
                val_file.write(sign + "\n")
                val_file.flush()


if __name__ == '__main__':
    main()
