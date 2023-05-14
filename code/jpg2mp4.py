import cv2
import os

def images_to_video(image_folder, output_path, fps):
    # 读取文件夹中的图像
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # 获取图像的宽度和高度
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 将图像写入视频文件
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放资源
    video.release()

# 指定图像文件夹路径、输出视频路径和帧率
image_folder = "vot2016/helicopter"
output_path = "video1.mp4"
fps = 30

# 调用函数将图像序列转换为视频
images_to_video(image_folder, output_path, fps)
