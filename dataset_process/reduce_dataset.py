import os


def main():
    all_csv_path = r'D:\document\service outsourcing\testdataset\all.csv'
    remove_labels = {'Pushing Two Fingers Away', 'Shaking Hand', 'Thumb Up', 'Doing other things'}  # 要删除的标签
    num = 350  # 每个标签要删除的数量
    i0 = 1
    i1 = 1
    i2 = 1
    i3 = 1
    with open(all_csv_path, 'r') as r:
        lines = r.readlines()
    with open(all_csv_path, 'w') as f:
        for line in lines:
            line_list = line.strip('\n').split(';')
            if line_list[1] not in remove_labels:
                f.write(line)
            elif line_list[1] == 'Pushing Two Fingers Away':
                if i0 <= num:
                    f.write(line)
                    i0 = i0 + 1
            elif line_list[1] == 'Shaking Hand':
                if i1 <= num:
                    f.write(line)
                    i1 = i1 + 1
            elif line_list[1] == 'Thumb Up':
                if i2 <= num:
                    f.write(line)
                    i2 = i2 + 1
            elif line_list[1] == 'Doing other things':
                if i3 <= num:
                    f.write(line)
                    i3 = i3 + 1


if __name__ == '__main__':
    main()
