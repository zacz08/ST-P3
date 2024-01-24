import cv2
import glob
import os

def to_video(folder_path, out_path, fps=4, downsample=1):
        imgs_path = [folder_path + "/" + img for img in os.listdir(folder_path)]
        # print("imgs_path:", imgs_path)
        imgs_path = sorted(imgs_path)
        img_array = []
        # print("imgs_path:", imgs_path)
        count =0
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height //
                             downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

def main():
    folder_path = '/home/zc/ST-P3/carla_sim_result/routes_devtest_01_14_21_58_22/show'
    # folder_path = 'ST-P3/carla_sim_result/routes_devtest_01_14_21_58_22/show'
    out_path = '/home/zc/Videos/stp3_car_sim_demo.avi'
    # vide_name = 'stp3_car_sim_demo.avi'
    to_video(folder_path, out_path)

if __name__=="__main__":
     main()