from PIL import Image
import os


def main():
    """
    左右翻转catch、clickDown、clickUp
    并添加至原数据集和all.csv之后
    """

    rgb_dir_path = r'D:\document\service outsourcing\testdataset\rgb'
    save_path = r'D:\document\service outsourcing\testdataset\rgb2'
    rgb_i = 2130
    save_i = 13171
    f = open(r'D:\document\service outsourcing\testdataset\all2.csv', 'a')
    while True:
        if rgb_i > 2860:
            break
        rgb_list = os.listdir(os.path.join(rgb_dir_path, str(rgb_i)))
        save_list = os.path.join(save_path, str(save_i))
        os.mkdir(save_list)
        for image_name in rgb_list:
            image1 = Image.open(os.path.join(rgb_dir_path, str(rgb_i), image_name))
            image2 = image1.transpose(Image.FLIP_LEFT_RIGHT)
            image2.save(save_list + '\\' + image_name)
        rgb_i = rgb_i + 1
        f.write(str(save_i) + ';' + 'Click Up\n')
        save_i = save_i + 1
    f.close()


if __name__ == '__main__':
    main()
