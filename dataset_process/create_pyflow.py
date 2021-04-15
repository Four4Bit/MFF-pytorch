import cv2
import os


def main():
    scr_path = r'D:\document\service outsourcing\pydataset\rgb'
    rgb_dir_list = os.listdir(scr_path)  # 存储rgb下的文件夹名字，1、2、3、4、5……
    dst_path = r'D:\document\service outsourcing\pydataset'
    flow_u_dir = os.path.join(dst_path, r'flow\u')
    flow_v_dir = os.path.join(dst_path, r'flow\v')
    if 'flow' not in os.listdir(dst_path):
        os.makedirs(flow_u_dir)
        os.makedirs(flow_v_dir)
    inst = cv2.optflow.createOptFlow_DeepFlow()
    for list_name in rgb_dir_list:
        rgb_picture_list_path = os.path.join(scr_path, list_name)
        rgb_picture_list = os.listdir(rgb_picture_list_path)  # 存储1文件夹下所有的图片的名字
        gray_list = []
        for rgb_picture_name in rgb_picture_list:
            rgb_picture_name_path = os.path.join(rgb_picture_list_path, rgb_picture_name)
            gray_list.append(cv2.imread(rgb_picture_name_path, 0))  # 0表示灰度图，存储所有rgb转换的灰度图
        for i in range(len(gray_list) - 1):
            frame1 = gray_list[i]
            frame2 = gray_list[i + 1]
            flow = inst.calc(frame1, frame2, None)
            # print(flow[:, :, 0])
            # print(len(flow[:, :, 0]))
            # print(flow[0, :, 0])
            # print(len(flow[0, :, 0]))
            # print("{:.20f}".format(flow[0, 0, 0])) 0.00171450036577880383
            # os.exit()
            if list_name not in os.listdir(flow_u_dir):
                os.makedirs(os.path.join(flow_u_dir, list_name))
                os.makedirs(os.path.join(flow_v_dir, list_name))
            flow_u_picture_path = os.path.join(flow_u_dir, list_name, "%05d.txt" % (i+1))
            flow_v_picture_path = os.path.join(flow_v_dir, list_name, "%05d.txt" % (i+1))
            print(flow_u_picture_path)
            print(flow_v_picture_path)  # 用于显示程序运行状态

            txt_u = open(flow_u_picture_path, 'a')
            for j in range(len(flow[:, :, 0])):
                for k in range(len(flow[j, :, 0])):
                    txt_u.write(str(format_change(flow[j, k, 0])) + ' ')
                txt_u.write('\n')
            txt_u.close()

            txt_v = open(flow_v_picture_path, 'a')
            for j in range(len(flow[:, :, 1])):
                for k in range(len(flow[j, :, 1])):
                    txt_v.write(str(format_change(flow[j, k, 1])) + ' ')
                txt_v.write('\n')
            txt_v.close()


def format_change(f):
    return "{:.12f}".format(f)


if __name__ == '__main__':
    main()
