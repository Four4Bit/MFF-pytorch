import os
import shutil

# target_path csv文件夹名
# file_path 手势文件所在文件夹
# label_name 标签名
# num 标签需要的数量
# list[] 标签对应的文件夹
# gesture_path 自己手势的目录

target_path = r'D:\Program Files\baidu\BaiduNetdiskDownload\Jester V1\jester-v1\jester-v1-train.csv'
file_path = r'D:\Program Files\baidu\BaiduNetdiskDownload\Jester V1\20bn-jester-v1'
label_name = 'Zooming Out With Two Fingers'
num = 25
list_l = []
gesture_path = r'D:\document\service outsourcing\dataset\rgb'


def main():
    search_files()
    copy_files()
    merge_files()


def search_files():
    i = 0
    f = open(target_path, 'r')
    while True:
        str_s = f.readline()
        if str_s == "":
            break
        str_s = str_s.split(';')
        # str_s[1].replace("\n", "")
        if str_s[1] == label_name+'\n':
            list_l.append(str_s[0])
            i += 1
        if i == num:
            break
    f.close()


def copy_files():
    i = 11079
    for name in list_l:
        oldpath = os.path.join(file_path, name)
        newpath = r'D:\Program Files\baidu\BaiduNetdiskDownload\Jester V1\new'+'\\'+str(i)
        shutil.copytree(oldpath, newpath)
        i += 1


def merge_files():
    f = open(r'D:\document\service outsourcing\dataset\all.csv', 'a')
    list_all = os.listdir(r'D:\Program Files\baidu\BaiduNetdiskDownload\Jester V1\new')
    for name in list_all:
        oldpath = r'D:\Program Files\baidu\BaiduNetdiskDownload\Jester V1\new'+'\\'+name
        newpath = gesture_path+'\\'+name
        shutil.copytree(oldpath, newpath)
        f.write(name+';'+'Doing other things'+'\n')
    f.close()


if __name__ == '__main__':
    main()
