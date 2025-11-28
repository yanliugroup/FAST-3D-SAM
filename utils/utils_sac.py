import torch
import numpy as np
import random


def get_ellipsoid_mask(center_pos, result_shape, mask_shape):
    """
    创建一个形状为[128, 128, 128]的 PyTorch Tensor,
    其中每个体素都满足以下条件:
    如果体素被以 center_pos 为中心、d_x、d_y、d_z 为偏心半径的椭球体内,则为 1,否则为 0.
    
    参数:
    center_pos (torch.Tensor): 形状为[3]的张量,表示椭球体的中心位置。
    d_x (float): 椭球体在 x 轴方向的半径。
    d_y (float): 椭球体在 y 轴方向的半径。
    d_z (float): 椭球体在 z 轴方向的半径。
    
    返回:
    torch.Tensor: 形状为[128, 128, 128]的张量,每个元素为 0 或 1。
    """
    d_x, d_y, d_z = mask_shape
    
    # 创建 x, y, z 坐标张量
    x = torch.arange(result_shape[0], dtype=torch.float32)
    y = torch.arange(result_shape[1], dtype=torch.float32)
    z = torch.arange(result_shape[2], dtype=torch.float32)
    
    # 将坐标张量广播到 [128, 128, 128] 的形状
    x, y, z = torch.meshgrid(x, y, z)
    
    # 计算每个体素到中心位置的距离
    dist_x = (x - center_pos[0]) ** 2 / (d_x ** 2)
    dist_y = (y - center_pos[1]) ** 2 / (d_y ** 2)
    dist_z = (z - center_pos[2]) ** 2 / (d_z ** 2)
    
    # 判断每个体素是否在椭球体内
    mask = (dist_x + dist_y + dist_z) <= 1
    
    return mask.to(torch.float32)


def compute_center_and_distances(points):
    """
    计算 12 个空间点的中心点和每个点到中心点在 x、y、z 方向上的最大距离。
    
    参数:
    points (torch.Tensor): 形状为 [12, 3] 的张量,代表 12 个空间点的坐标。
    
    返回:
    center (torch.Tensor): 形状为 [3] 的张量,代表中心点的坐标。
    max_distances (torch.Tensor): 形状为 [3] 的张量,代表每个点到中心点在 x、y、z 方向上的最大距离。
    """
    # 计算中心点
    center = points.mean(dim=0)
    
    # 计算每个点到中心点的距离
    distances = points - center.unsqueeze(0)
    
    # 计算每个方向上的最大距离
    max_distances = torch.max(torch.abs(distances), dim=0).values
    
    return center, max_distances

def get_saclick(seg):
    click_num = 12
    result_shape = seg.shape[1:]
    sa_segs = []
    for _seg in seg:
        _seg = _seg.unsqueeze(0)
        
        l = len(torch.where(_seg == 1)[0])
        if l > 0:
            sample = np.random.choice(np.arange(l), click_num, replace=True)
            x = torch.where(_seg == 1)[1][sample].unsqueeze(1)
            y = torch.where(_seg == 1)[2][sample].unsqueeze(1)
            z = torch.where(_seg == 1)[3][sample].unsqueeze(1)
            points_pos = torch.cat([x, y, z], dim=1).float()
            print("points_pos", points_pos)
            center_pos, mask_shape = compute_center_and_distances(points_pos)
            this_sa_seg = get_ellipsoid_mask(center_pos, result_shape, mask_shape).unsqueeze(0)
        else:
            print("no target")
            this_sa_seg = torch.zeros(result_shape).unsqueeze(0)
        sa_segs.append(this_sa_seg)
    
    return torch.cat(sa_segs, dim=0)
