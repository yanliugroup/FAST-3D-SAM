import pickle
import json
import os
import random
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

def adjust_window(image, window_width = 1400, window_level = 400):
    # 计算窗口的上下界
    window_min = window_level - window_width / 2
    window_max = window_level + window_width / 2
    
    # 将图像灰度值限制在窗口范围内
    adjusted_image = np.clip(image, window_min, window_max)
    
    # 将灰度值重新映射到0-1范围
    adjusted_image = (adjusted_image - window_min) / (window_max - window_min)
    
    return adjusted_image

def visualize_CT_mask(ct, mask, name):

    ct = adjust_window(ct)
    ct = ct * 255
    
    # 创建一个新的图形，并设置子图布局为 1 行 2 列
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # 可视化矩阵 A 在第一个子图中
    axs[0].imshow(ct, cmap='gray')

    # 叠加矩阵 B 在矩阵 A 上面，并设置透明度为 0.5
    axs[1].imshow(ct, cmap='gray')

    axs[1].imshow(mask, alpha=0.5)
    
    
    # 设置子图标题
    axs[0].set_title('CT')
    axs[1].set_title('CT & Mask')
    plt.title(name)
    
    # 调整子图间距
    plt.tight_layout()

    plt.savefig(f"exp_images/_find_multi_{name}.png", dpi=120)
    # 显示图形
    plt.clf()


dataset_root_folder = r"/home/qingzhongfei/sam/3DSAM-adapter/datafile/lung_hospital"

images = [p for p in os.listdir(os.path.join(dataset_root_folder, "imagesTr"))]
labels = [p for p in os.listdir(os.path.join(dataset_root_folder, "labelsTr"))]

for image, label in zip(images, labels):
    img_data = torch.load(os.path.join(dataset_root_folder, "imagesTr", image))
    label_data = torch.load(os.path.join(dataset_root_folder, "labelsTr", label))

    idx = 50
    visualize_CT_mask(img_data[idx], label_data[idx], image)
    break