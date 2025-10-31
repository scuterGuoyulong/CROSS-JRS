import math
import numpy as np
import torch.nn.functional as F
import torch
from monai.metrics import HausdorffDistanceMetric,compute_average_surface_distance
from natsort.compat.fake_fastnumbers import NAN_INF
from torch import nn
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F

import numpy as np
import nibabel as nib

import numpy as np
import nibabel as nib
import torch
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def process_deformation_field(deformation_tensor):
    """
    处理变形场张量并转换为numpy数组
    输入: (1, 3, 128, 128, 32) 的CUDA张量
    """
    # 转换到CPU并移除batch维度
    deformation = deformation_tensor.squeeze(0).cpu().numpy()  # (3, 128, 128, 32)

    # 重新排列维度为 (128, 128, 32, 3)
    deformation = np.transpose(deformation, (1, 2, 3, 0))

    return deformation


def create_2d_slice_visualization(deformation_field, slice_idx=16):
    """
    创建类似第一张图的2D切片3D可视化
    """
    # 选择一个Z切片
    slice_data = deformation_field[:, :, slice_idx, :]  # (128, 128, 3)

    # 计算位移大小
    magnitude = np.sqrt(np.sum(slice_data ** 2, axis=2))

    # 创建网格
    x = np.arange(128)
    y = np.arange(128)
    X, Y = np.meshgrid(x, y)

    # 缩放因子（调整Z轴高度）
    scale_factor = 20.0
    Z = magnitude * scale_factor

    # 创建3D表面图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制表面
    surf = ax.plot_surface(X, Y, Z, cmap='rainbow', alpha=0.8,
                           linewidth=0, antialiased=True)

    # 设置视角和标签
    ax.view_init(elev=30, azim=45)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Deformation Magnitude')
    ax.set_title(f'3D Deformation Field Visualization (Z-slice {slice_idx})')

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig


def create_interactive_3d_plot(deformation_field, slice_idx=16):
    """
    使用Plotly创建交互式3D可视化
    """
    # 选择一个Z切片
    slice_data = deformation_field[:, :, slice_idx, :]
    magnitude = np.sqrt(np.sum(slice_data ** 2, axis=2))

    # 创建网格
    x = np.arange(128)
    y = np.arange(128)
    X, Y = np.meshgrid(x, y)

    # 缩放因子
    scale_factor = 20.0
    Z = magnitude * scale_factor

    # 创建3D表面
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Rainbow',
        opacity=0.8,
        showscale=True
    )])

    fig.update_layout(
        title=f'Interactive 3D Deformation Field (Z-slice {slice_idx})',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Deformation Magnitude',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600
    )

    return fig


def create_multiple_slices_view(deformation_field, num_slices=4):
    """
    创建多个切片的3D可视化
    """
    slice_indices = np.linspace(5, 27, num_slices, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    for i, slice_idx in enumerate(slice_indices):
        slice_data = deformation_field[:, :, slice_idx, :]
        magnitude = np.sqrt(np.sum(slice_data ** 2, axis=2))

        x = np.arange(128)
        y = np.arange(128)
        X, Y = np.meshgrid(x, y)

        scale_factor = 20.0
        Z = magnitude * scale_factor

        surf = axes[i].plot_surface(X, Y, Z, cmap='rainbow', alpha=0.8,
                                    linewidth=0, antialiased=True)

        axes[i].view_init(elev=30, azim=45)
        axes[i].set_title(f'Z-slice {slice_idx}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_zlabel('Magnitude')

    plt.tight_layout()
    return fig


def create_vector_field_visualization(deformation_field, slice_idx=16, step=8):
    """
    创建向量场可视化
    """
    slice_data = deformation_field[:, :, slice_idx, :]
    magnitude = np.sqrt(np.sum(slice_data ** 2, axis=2))

    # 子采样以避免过于密集的箭头
    x = np.arange(0, 128, step)
    y = np.arange(0, 128, step)
    X, Y = np.meshgrid(x, y)

    # 获取子采样的向量
    U = slice_data[::step, ::step, 0]
    V = slice_data[::step, ::step, 1]

    # 创建基础高度图
    scale_factor = 20.0
    Z_base = magnitude * scale_factor

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制表面
    x_full = np.arange(128)
    y_full = np.arange(128)
    X_full, Y_full = np.meshgrid(x_full, y_full)
    surf = ax.plot_surface(X_full, Y_full, Z_base, cmap='rainbow', alpha=0.6)

    # 添加向量场箭头
    for i in range(len(x)):
        for j in range(len(y)):
            ax.quiver(X[j, i], Y[j, i], Z_base[y[j], x[i]],
                      U[j, i] * 10, V[j, i] * 10, 0,
                      color='black', alpha=0.7, arrow_length_ratio=0.3)

    ax.view_init(elev=30, azim=45)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Deformation Magnitude')
    ax.set_title('3D Deformation Field with Vector Arrows')

    return fig


import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image  # 新增：用于图像裁剪

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


def save_deformation_field_as_rgb(flow,
                                  save_dir='./',
                                  title='deformation_field_rgb.png',
                                  dpi=300):
    """
    将变形场保存为二维 RGB 图像（不进行 3D 视角调整，直接提取切片保存）

    :param flow: PyTorch 张量，形状应为 (1, 3, H, W, D) 或适配的 5D 变形场张量
    :param save_dir: 图像保存的目录路径
    :param filename: 保存的图像文件名
    :param dpi: 保存图像的分辨率
    """
    try:
        # 输入校验
        if flow.dim() != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
            raise ValueError("flow 张量形状应为 (1, 3, H, W, D)")

        # 提取中间切片（沿深度维度 D 取中间）
        slice_idx = flow.size(-1) // 2
        flow_slice = flow[0, :, :, :, slice_idx].cpu().numpy()

        # 调整维度顺序并归一化到 [0, 1] 范围，用于 RGB 显示
        flow_rgb = flow_slice.transpose(1, 2, 0)
        flow_rgb = (flow_rgb - flow_rgb.min()) / (flow_rgb.max() - flow_rgb.min())
        flow_rgb=np.rot90(flow_rgb, k=1)
        # 创建画布并绘制 RGB 图像
        fig = plt.figure(figsize=(12, 12), dpi=180)
        ax = fig.add_axes([0, 0, 1, 1])  # 让图像占满整个画布，去除边距影响
        ax.imshow(flow_rgb)
        ax.axis('off')  # 隐藏坐标轴

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, title)

        # 保存图像（去除白边，背景透明可按需调整）
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0,
            transparent=False  # 若需要透明背景可设为 True，结合后续处理
        )
        plt.close()

        print(f"变形场已保存为二维 RGB 图像：{save_path}")

    except Exception as e:
        print(f"保存变形场为 RGB 图像时发生错误: {e}")
        raise
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def visualize_and_crop_png(image_path,
                           fig_size=(6, 6),
                           elev=20,
                           azim=-60,
                           bg_color='none',
                           save_dir='./',
                           title='output'):
    """
    对 PNG 图片进行视角调整（模拟 3D 视角倾斜效果）和裁剪空白区域
    :param image_path: PNG 图片路径
    :param fig_size: 画布尺寸，tuple 类型，如 (6, 6)
    :param elev: 3D 视角仰角，用于调整视角
    :param azim: 3D 视角方位角，用于调整视角倾斜效果
    :param bg_color: 画布及子图背景颜色，'none' 表示透明
    :param save_dir: 结果保存目录
    :param title: 输出图片文件名（不含扩展名）
    """
    # 1. 加载图片并转换为 RGBA 格式
    img = Image.open(image_path).convert("RGBA")
    img_data = np.array(img)
    height, width, _ = img_data.shape

    # 2. 创建 3D 画布 + 透明背景
    fig = plt.figure(figsize=fig_size, facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg_color)

    # 3. 生成平面网格，将 2D 图片映射到 3D 平面
    y = np.linspace(-0.2, 0.2, width)
    z = np.linspace(-1, 1, height)
    Y, Z = np.meshgrid(y, z)
    X = np.full_like(Y, 0)  # 让图片处于 x=0 平面

    # 处理图片数据为 3D 平面的 facecolors 可用格式
    # 提取 RGB 通道，归一化到 0-1 范围
    img_rgb = img_data[:, :, :3] / 255.0
    # 重复为 (height, width, 3) ，与网格维度匹配
    facecolors = np.broadcast_to(img_rgb, (height, width, 3))

    # 4. 绘制带纹理的平面
    surf = ax.plot_surface(
        X, Y, Z,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        shade=False
    )

    # 5. 视角调整
    ax.view_init(elev=elev, azim=azim)

    # 6. 隐藏所有轴线、刻度、背景等
    ax.set_axis_off()
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # 7. 调整布局，保存临时图像
    plt.tight_layout(pad=0)
    temp_path = f"{save_dir}/temp_{title}.png"
    plt.savefig(
        temp_path,
        transparent=True,
        bbox_inches='tight',
        pad_inches=0,
        dpi=300
    )
    plt.close()

    # 8. 精准裁剪空白（基于透明通道）
    img_temp = Image.open(temp_path).convert("RGBA")
    data_temp = np.array(img_temp)
    alpha = data_temp[:, :, 3]

    non_empty_rows = np.where(alpha != 0)[0]
    non_empty_cols = np.where(alpha != 0)[1]

    if len(non_empty_rows) == 0 or len(non_empty_cols) == 0:
        img_temp.save(f"{save_dir}/{title}.png")
        os.remove(temp_path)
        return

    min_row, max_row = non_empty_rows.min(), non_empty_rows.max()
    min_col, max_col = non_empty_cols.min(), non_empty_cols.max()

    cropped_data = data_temp[min_row:max_row + 1, min_col:max_col + 1]
    cropped_img = Image.fromarray(cropped_data)

    # 9. 保存最终裁剪后的图像并删除临时文件
    cropped_img.save(f"{save_dir}/{title}.png")
    os.remove(temp_path)

def visualize_deformation_field(flow, slice_idx=16,
                                fig_size=(6, 6),
                                elev=20, azim=-60,  # 调整方位角实现垂直倾斜
                                label=r'$\phi$',
                                bg_color='none',save_dir='./',title='flow'):    # 背景透明参数
    """
    改造：支持垂直视角倾斜 + 透明背景
    :param azim: 重点调整方位角，实现类似“竖板倾斜”视角
    :param bg_color: 画布背景颜色，'none' 表示透明
    """
    # 旋转一下
    flow=flow.permute(0, 1, 3,2, 4)
    # 1. 输入校验与切片提取（同之前）
    if flow.dim() != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
        raise ValueError("flow 张量形状应为 (1, 3, H, W, D)")
    slice_idx=flow.size(-1)//2
    phi_slice = flow[0, :, :, :, slice_idx].cpu().numpy()
    phi_rgb = phi_slice.transpose(1, 2, 0)
    phi_rgb = (phi_rgb - phi_rgb.min()) / (phi_rgb.max() - phi_rgb.min())

    # 2. 创建 3D 画布 + 透明背景
    fig = plt.figure(figsize=fig_size, facecolor=bg_color)  # 画布背景透明
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg_color)  # 子图背景也透明

    # 3. 生成平面网格（适配垂直倾斜）
    size = phi_rgb.shape[0]
    # 调整轴范围，让立体效果更贴近参考图
    y = np.linspace(-0.2, 0.2, size)
    z = np.linspace(-1, 1, size)
    Y, Z = np.meshgrid(y, z)
    X = np.full_like(Y, 0)  # 平面位置

    # 4. 绘制带纹理的平面
    surf = ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        facecolors=phi_rgb,
        shade=False
    )

    # 5. 视角调整（关键：用 azim=-60 实现垂直倾斜）
    ax.view_init(elev=elev, azim=azim)

    # 6. 隐藏所有轴线、刻度、背景
    ax.set_axis_off()  # 隐藏坐标轴
    # 额外隐藏 3D 背景网格（若有）
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # 7. 调整布局 + 透明显示
    plt.tight_layout(pad=0)  # 去除边距
    # 8. 保存临时图像（含透明背景）
    temp_path = f"{save_dir}/temp_{title}.png"
    plt.savefig(
        temp_path,
        transparent=True,
        bbox_inches='tight',  # 初步裁剪
        pad_inches=0,  # 边距设为 0
        dpi=300  # 高分辨率，避免裁剪损失
    )
    plt.close()

    # 9. 精准裁剪空白（核心步骤）
    # 打开临时图像，处理透明通道
    img = Image.open(temp_path).convert("RGBA")
    data = np.array(img)
    alpha = data[:, :, 3]  # 提取透明通道

    # 找到非透明像素的最小包围盒
    non_empty_rows = np.where(alpha != 0)[0]
    non_empty_cols = np.where(alpha != 0)[1]

    if len(non_empty_rows) == 0 or len(non_empty_cols) == 0:
        # 全透明时直接保存临时文件（避免报错）
        img.save(f"{save_dir}/{title}.png")
        return

    # 计算裁剪边界
    min_row, max_row = non_empty_rows.min(), non_empty_rows.max()
    min_col, max_col = non_empty_cols.min(), non_empty_cols.max()

    # 执行裁剪
    cropped_data = data[min_row:max_row + 1, min_col:max_col + 1]
    cropped_img = Image.fromarray(cropped_data)

    # 10. 保存最终裁剪后的图像
    cropped_img.save(f"{save_dir}/{title}.png")

    # 11. （可选）删除临时文件
    import os
    os.remove(temp_path)


import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



def visualize_gray_img(flow, slice_idx=16,
                       fig_size=(6, 6),
                       elev=20, azim=-60,
                       label=r'$\phi$',
                       bg_color='none',
                       cmap='gray',
                       save_dir='./',
                       title='flow'):
    # 旋转处理
    flow = flow.permute(0, 1, 3, 2, 4)

    # 1. 输入校验与切片提取
    if flow.dim() != 5 or flow.shape[0] != 1 or flow.shape[1] != 1:
        raise ValueError("flow 张量形状应为 (1, 1, H, W, D)")

    phi_slice = flow[0, 0, :, :, slice_idx].cpu().numpy()

    # 2. 归一化到 [0,1]
    phi_norm = (phi_slice - phi_slice.min()) / (phi_slice.max() - phi_slice.min())

    # 3. 创建 3D 画布 + 透明背景
    fig = plt.figure(figsize=fig_size, facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg_color)

    # 4. 生成平面网格
    size = phi_norm.shape[0]
    y = np.linspace(-0.2, 0.2, size)
    z = np.linspace(-1, 1, size)
    Y, Z = np.meshgrid(y, z)
    X = np.full_like(Y, 0)

    # 5. 灰度转 RGB
    phi_rgb = plt.cm.get_cmap(cmap)(phi_norm)[:, :, :3]

    # 6. 绘制平面
    ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        facecolors=phi_rgb,
        shade=False
    )

    # 7. 视角与坐标轴设置
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # 8. 保存临时图像（含透明背景）
    temp_path = f"{save_dir}/temp_{title}.png"
    plt.savefig(
        temp_path,
        transparent=True,
        bbox_inches='tight',  # 初步裁剪
        pad_inches=0,  # 边距设为 0
        dpi=300  # 高分辨率，避免裁剪损失
    )
    plt.close()

    # 9. 精准裁剪空白（核心步骤）
    # 打开临时图像，处理透明通道
    img = Image.open(temp_path).convert("RGBA")
    data = np.array(img)
    alpha = data[:, :, 3]  # 提取透明通道

    # 找到非透明像素的最小包围盒
    non_empty_rows = np.where(alpha != 0)[0]
    non_empty_cols = np.where(alpha != 0)[1]

    if len(non_empty_rows) == 0 or len(non_empty_cols) == 0:
        # 全透明时直接保存临时文件（避免报错）
        img.save(f"{save_dir}/{title}.png")
        return

    # 计算裁剪边界
    min_row, max_row = non_empty_rows.min(), non_empty_rows.max()
    min_col, max_col = non_empty_cols.min(), non_empty_cols.max()

    # 执行裁剪
    cropped_data = data[min_row:max_row + 1, min_col:max_col + 1]
    cropped_img = Image.fromarray(cropped_data)

    # 10. 保存最终裁剪后的图像
    cropped_img.save(f"{save_dir}/{title}.png")

    # 11. （可选）删除临时文件
    import os
    os.remove(temp_path)





def Dice(vol1, vol2, labels=None, nargout=1):
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return np.mean(dicem)
    else:
        return (dicem, labels)

def downsample_labels(labels):
    """
    等比例缩小分割标签。

    参数:
    labels (torch.Tensor): 输入的分割标签，形状为 (batch_size, channels, depth, height, width)

    返回:
    list: 缩小后的分割标签列表
    """
    scale_factor = [0.5, 0.25, 0.125]
    assert all(s in [0.5, 0.25, 0.125] for s in scale_factor), "仅支持 0.5, 0.25, 0.125 的缩放比例"

    downsampled_labels = []
    # 使用 trilinear 插值方法进行下采样
    for s in scale_factor:
        # 确保 scale_factor 是正确的格式
        s = (s, s, s)
        downsampled_label = F.interpolate(labels.float(), scale_factor=s, mode='trilinear', align_corners=True)
        # 将结果转换回整数类型
        downsampled_labels.append(downsampled_label.long())

    return downsampled_labels

def read_txt_landmarks(filename, if_down=False):
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    landmarks = []
    for line in Lines:
        line = line.strip().split('\t')
        #print("Line: {}".format(line))
        if if_down:
            lm = [int(int(line[0])/2), int(int(line[1])/2), int(line[2])]
        else:
            lm = [int(line[0]), int(line[1]), int(line[2])]
        landmarks.append(lm)
    return landmarks

def deform_landmarks(source_lms, target_lms, flow, if_down=False):
    factor = 1
    if if_down:
        factor = 2
    u = flow[0, :, :, :]
    v = flow[1, :, :, :]
    w = flow[2, :, :, :]
    pixel_wth = [2.5, 0.97*factor, 0.97*factor]
    flow_fields = [u, v, w]
    diff_all = []
    raw_diff_all = []
    for i in range(len(source_lms)):
        source_lm = source_lms[i]
        target_lm = target_lms[i]
        diff = 0
        raw_diff = 0
        for j in range(len(source_lm)):
            sor_pnt = source_lm[2-j]
            tar_pnt = target_lm[2-j]
            def_field = flow_fields[j]
            out_pnt = sor_pnt - def_field[source_lm[2]-1, source_lm[1]-1, source_lm[0]-1]
            diff += (np.abs(out_pnt-tar_pnt)*pixel_wth[j])**2
            raw_diff += (np.abs(sor_pnt-tar_pnt)*pixel_wth[j])**2
        diff_all.append(math.sqrt(diff))
        raw_diff_all.append(math.sqrt(raw_diff))
    return np.mean(np.array(diff_all)), np.mean(np.std(diff_all)), np.mean(np.array(raw_diff_all)), np.std(np.array(raw_diff_all))

def dice_val(y_pred, y_true):
    # 将 y_pred 转换为 Tensor
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # 确保 y_pred 和 y_true 在同一设备上
    if y_pred.device != y_true.device:
        y_pred = y_pred.to(y_true.device)
    intersection = (y_pred * y_true).sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2. * intersection) / (union + 1e-5)
    return torch.mean(dsc)

def dice_val_VOI(y_pred, y_true):
    VOI_lbls = [0,1, 2, 3, 5, 6]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] =dsc
        idx += 1
    return 1-np.mean(DSCs)

def dice_val_substruct(y_pred, y_true, std_idx):
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=46)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=46)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(1,2):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line

def dice_val_CTsubstruct(y_pred, y_true, std_idx):
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=16)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=16)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(16):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line

import re
def process_label():
    seg_table = [0, 1,2]


    file1 = open('label_info.txt', 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ',line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict

def process_CT_label():
    seg_table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    seg_name = ['Body-Outline', 'Bone-Structure', 'Right-Lung', 'Left-Lung', 'Heart', 'Liver', 'Spleen', 'Right-Kidney',
                'Left-Kidney', 'Stomach', 'Pancreas', 'Large-Intestine', 'Prostate', 'Bladder', 'Gall-Bladder', 'Thyroid']
    return seg_name

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[1:]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, 0)
    #print(grid)
    #sys.exit(0)

    # compute gradients
    [xFX, xFY, xFZ] = np.gradient(grid[0] - disp[0])
    [yFX, yFY, yFZ] = np.gradient(grid[1] - disp[1])
    [zFX, zFY, zFZ] = np.gradient(grid[2] - disp[2])

    jac_det = np.zeros(grid[0].shape)
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            for k in range(grid.shape[3]):
                jac_mij = [[xFX[i, j, k], xFY[i, j, k], xFZ[i, j, k]], [yFX[i, j, k], yFY[i, j, k], yFZ[i, j, k]], [zFX[i, j, k], zFY[i, j, k], zFZ[i, j, k]]]
                jac_det[i, j, k] =  np.linalg.det(jac_mij)

    # 3D glow
    #if nb_dims == 3:
    #    dx = J[0]
    #    dy = J[1]
    #    dz = J[2]

        # compute jacobian components
    #    Jdet0 = dx[0, ...] * (dy[1, ...] * dz[2, ...] - dy[2, ...] * dz[1, ...])
    #    Jdet1 = dx[1, ...] * (dy[0, ...] * dz[2, ...] - dy[2, ...] * dz[0, ...])
    #    Jdet2 = dx[2, ...] * (dy[0, ...] * dz[1, ...] - dy[1, ...] * dz[0, ...])

    #    return Jdet0 - Jdet1 + Jdet2

    #else:  # must be 2

    #    dfdx = J[0]
    #    dfdy = J[1]

    #    return dfdx[0, ...] * dfdy[1, ...] - dfdy[0, ...] * dfdx[1, ...]
    return jac_det

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def dice(y_pred, y_true, ):
    intersection = y_pred * y_true
    intersection = np.sum(intersection)
    union = np.sum(y_pred) + np.sum(y_true)
    dsc = (2.*intersection) / (union + 1e-5)
    return dsc

def smooth_seg(binary_img, sigma=1.5, thresh=0.5):
    binary_img = gaussian_filter(binary_img.astype(np.float32()), sigma=sigma)
    binary_img = binary_img > thresh
    return binary_img

def checkboard(shape, block_sz = 20):
    sz = 20
    xvalue = 64
    yvalue = 64
    A = np.zeros((sz, sz))
    B = np.ones((sz, sz))
    C = np.zeros((sz*xvalue, sz*yvalue))
    m = sz
    n = 0
    num = 2
    for i in range(xvalue):
        n1=0
        m1=sz
        for j in range(yvalue):
            if num % 2 == 0:
                C[n:m, n1:m1] = A
                num += 1
            else:
                C[n:m, n1:m1] = B
                num += 1
            n1 = n1 + sz
            m1 = m1 + sz
        if yvalue%2 == 0:
            num = num + 1
        n = n + sz
        m = m + sz

    C = C[0:shape[0], 0:shape[1]]
    return C

def compute_tre(y_true, y_pred):
    # 找到 y_true 和 y_pred 中前景质心
    def compute_centroid(mask):
        centroids = []
        for i in range(mask.shape[0]):
            indices = torch.nonzero(mask[i] > 0.5, as_tuple=False)
            if indices.shape[0] > 0:
                centroid = torch.mean(indices.float(), dim=0)
            else:
                centroid = torch.zeros(4, device=mask.device)
            centroids.append(centroid)
        return torch.stack(centroids, dim=0)[:,1:]

    y_true_centroids = compute_centroid(y_true)
    # print(y_true_centroids)
    y_pred_centroids = compute_centroid(y_pred)
    # print(y_pred_centroids)
    # print(y_true_centroids)
    # print(y_pred_centroids)
    # 计算欧几里得距离
    distances = torch.norm(y_true_centroids - y_pred_centroids, dim=1)

    # 返回平均TRE损失
    return distances.item()
import numpy as np
import torch


def get_index(lmark_truth, lmark_pred, number):

    index_t = torch.where(lmark_truth == number)
    index_p = torch.where(lmark_pred == number)

  #  print(f" number: {number} \n index_t: {index_t}  \n index_p: {index_p}")
    #
    if len(index_p[0]) < 1:
        if torch.mean(index_t[0].float()) > 41:
            ind_x = 90
        else:
            ind_x = - 10
        if torch.mean(index_t[1].float()) > 41:
            ind_y = 90
        else:
            ind_y = - 10
        if torch.mean(index_t[2].float()) > 41:
            ind_z = 90
        else:
            ind_z = - 10
        index_p = [torch.tensor([ind_x]), torch.tensor([ind_y]), torch.tensor([ind_z])]
    return index_t, index_p


#

def get_distance_lmark(lmark_truth, lmark_pred, device):
    landmark_tot_distance = []


    for landmark in range(int(torch.max(lmark_truth))):
        index_t, index_p = get_index(lmark_truth[0,0 :, :, :], lmark_pred[0,0 :, :, :], 1)

        if len(index_t) != len(index_p):
            continue
        print(index_t, index_p)
        diff = torch.stack(index_t)[1:4, :].float().mean(dim=1) - torch.stack(index_p)[1:4, :].float().mean(dim=1)

        landmark_tot_distance.append((diff ** 2).sum().sqrt().to(device))

    if len(landmark_tot_distance) == 0:
        return torch.tensor([0.0]).to(device)
    else:
        return torch.mean(torch.stack(landmark_tot_distance))

def compute_seg_metric(y_pre,y_true):
    hd95 = HausdorffDistanceMetric(percentile=95.0)
    distance_hd95 = hd95(y_pre,y_true)

    distance_ASSD = compute_average_surface_distance(symmetric=True,y_pred=y_pre,y=y_true)
    tre=compute_tre(y_pre,y_true)

    hd95_max_value=30.0
    assd_max_value=30.0

    if torch.isnan(distance_hd95).any() or torch.isinf(distance_hd95).any() :
        distance_hd95 = torch.tensor(hd95_max_value)
    if torch.isnan(distance_ASSD).any() or torch.isinf(distance_ASSD).any() :
        distance_ASSD = torch.tensor(assd_max_value)
    # tre=get_distance_lmark(y_true,y_pre,y_pre.device)
    return distance_hd95.item(), distance_ASSD.item(), tre


class Dense3DSpatialTransformer(nn.Module):
    def __init__(self):
        super(Dense3DSpatialTransformer, self).__init__()

    def forward(self, input1, input2):
        return self._transform(input1, input2[:, 0], input2[:, 1], input2[:, 2])

    def _transform(self, input1, dDepth, dHeight, dWidth):
        batchSize = dDepth.shape[0]
        dpt = dDepth.shape[1]
        hgt = dDepth.shape[2]
        wdt = dDepth.shape[3]

        D_mesh, H_mesh, W_mesh = self._meshgrid(dpt, hgt, wdt)
        D_mesh = D_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        D_upmesh = dDepth + D_mesh
        H_upmesh = dHeight + H_mesh
        W_upmesh = dWidth + W_mesh

        return self._interpolate(input1, D_upmesh, H_upmesh, W_upmesh)

    def _meshgrid(self, dpt, hgt, wdt):
        d_t = torch.linspace(0.0, dpt-1.0, dpt).unsqueeze_(1).unsqueeze_(1).expand(dpt, hgt, wdt).cuda()
        h_t = torch.matmul(torch.linspace(0.0, hgt-1.0, hgt).unsqueeze_(1), torch.ones((1,wdt))).cuda()
        h_t = h_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        w_t = torch.matmul(torch.ones((hgt,1)), torch.linspace(0.0, wdt-1.0, wdt).unsqueeze_(1).transpose(1,0)).cuda()
        w_t = w_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        return d_t, h_t, w_t

    def _interpolate(self, input, D_upmesh, H_upmesh, W_upmesh):
        nbatch = input.shape[0]
        nch    = input.shape[1]
        depth  = input.shape[2]
        height = input.shape[3]
        width  = input.shape[4]

        img = torch.zeros(nbatch, nch, depth+2,  height+2, width+2).cuda()
        img[:, :, 1:-1, 1:-1, 1:-1] = input

        imgDpt = img.shape[2]
        imgHgt = img.shape[3]
        imgWdt = img.shape[4]

        # D_upmesh, H_upmesh, W_upmesh = [D, H, W] -> [BDHW,]
        D_upmesh = D_upmesh.view(-1).float()+1.0  # (BDHW,)
        H_upmesh = H_upmesh.view(-1).float()+1.0  # (BDHW,)
        W_upmesh = W_upmesh.view(-1).float()+1.0  # (BDHW,)

        # D_upmesh, H_upmesh, W_upmesh -> Clamping
        df = torch.floor(D_upmesh).int()
        dc = df + 1
        hf = torch.floor(H_upmesh).int()
        hc = hf + 1
        wf = torch.floor(W_upmesh).int()
        wc = wf + 1

        df = torch.clamp(df, 0, imgDpt-1)  # (BDHW,)
        dc = torch.clamp(dc, 0, imgDpt-1)  # (BDHW,)
        hf = torch.clamp(hf, 0, imgHgt-1)  # (BDHW,)
        hc = torch.clamp(hc, 0, imgHgt-1)  # (BDHW,)
        wf = torch.clamp(wf, 0, imgWdt-1)  # (BDHW,)
        wc = torch.clamp(wc, 0, imgWdt-1)  # (BDHW,)

        # Find batch indexes
        rep = torch.ones([depth*height*width, ]).unsqueeze_(1).transpose(1, 0).cuda()
        bDHW = torch.matmul((torch.arange(0, nbatch).float()*imgDpt*imgHgt*imgWdt).unsqueeze_(1).cuda(), rep).view(-1).int()

        # Box updated indexes
        HW = imgHgt*imgWdt
        W = imgWdt
        # x: W, y: H, z: D
        idx_000 = bDHW + df*HW + hf*W + wf
        idx_100 = bDHW + dc*HW + hf*W + wf
        idx_010 = bDHW + df*HW + hc*W + wf
        idx_110 = bDHW + dc*HW + hc*W + wf
        idx_001 = bDHW + df*HW + hf*W + wc
        idx_101 = bDHW + dc*HW + hf*W + wc
        idx_011 = bDHW + df*HW + hc*W + wc
        idx_111 = bDHW + dc*HW + hc*W + wc

        # Box values
        img_flat = img.view(-1, nch).float()  # (BDHW,C) //// C=1

        val_000 = torch.index_select(img_flat, 0, idx_000.long())
        val_100 = torch.index_select(img_flat, 0, idx_100.long())
        val_010 = torch.index_select(img_flat, 0, idx_010.long())
        val_110 = torch.index_select(img_flat, 0, idx_110.long())
        val_001 = torch.index_select(img_flat, 0, idx_001.long())
        val_101 = torch.index_select(img_flat, 0, idx_101.long())
        val_011 = torch.index_select(img_flat, 0, idx_011.long())
        val_111 = torch.index_select(img_flat, 0, idx_111.long())

        dDepth  = dc.float() - D_upmesh
        dHeight = hc.float() - H_upmesh
        dWidth  = wc.float() - W_upmesh

        wgt_000 = (dWidth*dHeight*dDepth).unsqueeze_(1)
        wgt_100 = (dWidth * dHeight * (1-dDepth)).unsqueeze_(1)
        wgt_010 = (dWidth * (1-dHeight) * dDepth).unsqueeze_(1)
        wgt_110 = (dWidth * (1-dHeight) * (1-dDepth)).unsqueeze_(1)
        wgt_001 = ((1-dWidth) * dHeight * dDepth).unsqueeze_(1)
        wgt_101 = ((1-dWidth) * dHeight * (1-dDepth)).unsqueeze_(1)
        wgt_011 = ((1-dWidth) * (1-dHeight) * dDepth).unsqueeze_(1)
        wgt_111 = ((1-dWidth) * (1-dHeight) * (1-dDepth)).unsqueeze_(1)

        output = (val_000*wgt_000 + val_100*wgt_100 + val_010*wgt_010 + val_110*wgt_110 +
                  val_001 * wgt_001 + val_101 * wgt_101 + val_011 * wgt_011 + val_111 * wgt_111)
        output = output.view(nbatch, depth, height, width, nch).permute(0, 4, 1, 2, 3)  #B, C, D, H, W
        return output

# Spatial Transformer 3D Net #################################################
class Dense3DSpatialTransformerNN(nn.Module):
    def __init__(self):
        super(Dense3DSpatialTransformerNN, self).__init__()

    def forward(self, input1, input2):
        return self._transform(input1, input2[:, 0], input2[:, 1], input2[:, 2])

    def _transform(self, input1, dDepth, dHeight, dWidth):
        batchSize = dDepth.shape[0]
        dpt = dDepth.shape[1]
        hgt = dDepth.shape[2]
        wdt = dDepth.shape[3]

        D_mesh, H_mesh, W_mesh = self._meshgrid(dpt, hgt, wdt)
        D_mesh = D_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        D_upmesh = dDepth + D_mesh
        H_upmesh = dHeight + H_mesh
        W_upmesh = dWidth + W_mesh

        return self._interpolate(input1, D_upmesh, H_upmesh, W_upmesh)

    def _meshgrid(self, dpt, hgt, wdt):
        d_t = torch.linspace(0.0, dpt-1.0, dpt).unsqueeze_(1).unsqueeze_(1).expand(dpt, hgt, wdt).cuda()
        h_t = torch.matmul(torch.linspace(0.0, hgt-1.0, hgt).unsqueeze_(1), torch.ones((1,wdt))).cuda()
        h_t = h_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        w_t = torch.matmul(torch.ones((hgt,1)), torch.linspace(0.0, wdt-1.0, wdt).unsqueeze_(1).transpose(1,0)).cuda()
        w_t = w_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        return d_t, h_t, w_t

    def _interpolate(self, input, D_upmesh, H_upmesh, W_upmesh):
        nbatch = input.shape[0]
        nch    = input.shape[1]
        depth  = input.shape[2]
        height = input.shape[3]
        width  = input.shape[4]

        img = torch.zeros(nbatch, nch, depth+2,  height+2, width+2).cuda()
        img[:, :, 1:-1, 1:-1, 1:-1] = input

        imgDpt = img.shape[2]
        imgHgt = img.shape[3]
        imgWdt = img.shape[4]

        # D_upmesh, H_upmesh, W_upmesh = [D, H, W] -> [BDHW,]
        D_upmesh = D_upmesh.view(-1).float()+1.0  # (BDHW,)
        H_upmesh = H_upmesh.view(-1).float()+1.0  # (BDHW,)
        W_upmesh = W_upmesh.view(-1).float()+1.0  # (BDHW,)

        # D_upmesh, H_upmesh, W_upmesh -> Clamping
        df = torch.floor(D_upmesh).int()
        dc = df + 1
        hf = torch.floor(H_upmesh).int()
        hc = hf + 1
        wf = torch.floor(W_upmesh).int()
        wc = wf + 1

        df = torch.clamp(df, 0, imgDpt-1)  # (BDHW,)
        dc = torch.clamp(dc, 0, imgDpt-1)  # (BDHW,)
        hf = torch.clamp(hf, 0, imgHgt-1)  # (BDHW,)
        hc = torch.clamp(hc, 0, imgHgt-1)  # (BDHW,)
        wf = torch.clamp(wf, 0, imgWdt-1)  # (BDHW,)
        wc = torch.clamp(wc, 0, imgWdt-1)  # (BDHW,)

        # Find batch indexes
        rep = torch.ones([depth*height*width, ]).unsqueeze_(1).transpose(1, 0).cuda()
        bDHW = torch.matmul((torch.arange(0, nbatch).float()*imgDpt*imgHgt*imgWdt).unsqueeze_(1).cuda(), rep).view(-1).int()

        # Box updated indexes
        HW = imgHgt*imgWdt
        W = imgWdt
        # x: W, y: H, z: D
        idx_000 = bDHW + df*HW + hf*W + wf
        idx_100 = bDHW + dc*HW + hf*W + wf
        idx_010 = bDHW + df*HW + hc*W + wf
        idx_110 = bDHW + dc*HW + hc*W + wf
        idx_001 = bDHW + df*HW + hf*W + wc
        idx_101 = bDHW + dc*HW + hf*W + wc
        idx_011 = bDHW + df*HW + hc*W + wc
        idx_111 = bDHW + dc*HW + hc*W + wc

        # Box values
        img_flat = img.view(-1, nch).float()  # (BDHW,C) //// C=1

        val_000 = torch.index_select(img_flat, 0, idx_000.long())
        val_100 = torch.index_select(img_flat, 0, idx_100.long())
        val_010 = torch.index_select(img_flat, 0, idx_010.long())
        val_110 = torch.index_select(img_flat, 0, idx_110.long())
        val_001 = torch.index_select(img_flat, 0, idx_001.long())
        val_101 = torch.index_select(img_flat, 0, idx_101.long())
        val_011 = torch.index_select(img_flat, 0, idx_011.long())
        val_111 = torch.index_select(img_flat, 0, idx_111.long())

        dDepth  = torch.round(dc.float() - D_upmesh)
        dHeight = torch.round(hc.float() - H_upmesh)
        dWidth  = torch.round(wc.float() - W_upmesh)

        wgt_000 = (dWidth*dHeight*dDepth).unsqueeze_(1)
        wgt_100 = (dWidth * dHeight * (1-dDepth)).unsqueeze_(1)
        wgt_010 = (dWidth * (1-dHeight) * dDepth).unsqueeze_(1)
        wgt_110 = (dWidth * (1-dHeight) * (1-dDepth)).unsqueeze_(1)
        wgt_001 = ((1-dWidth) * dHeight * dDepth).unsqueeze_(1)
        wgt_101 = ((1-dWidth) * dHeight * (1-dDepth)).unsqueeze_(1)
        wgt_011 = ((1-dWidth) * (1-dHeight) * dDepth).unsqueeze_(1)
        wgt_111 = ((1-dWidth) * (1-dHeight) * (1-dDepth)).unsqueeze_(1)

        output = (val_000*wgt_000 + val_100*wgt_100 + val_010*wgt_010 + val_110*wgt_110 +
                  val_001 * wgt_001 + val_101 * wgt_101 + val_011 * wgt_011 + val_111 * wgt_111)
        output = output.view(nbatch, depth, height, width, nch).permute(0, 4, 1, 2, 3)  #B, C, D, H, W
        return output

class register_model(nn.Module):
    def __init__(self,  img_sz=None, mode='bilinear'):
        super(register_model, self).__init__()
        if mode=='bilinear':
            self.spatial_trans = Dense3DSpatialTransformer()
        else:
            self.spatial_trans = Dense3DSpatialTransformerNN()

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def update_metric(metrics,metric,x,tar):

    eval_dsc_def,eval_dsc_raw,eval_hd95_def,eval_hd95_raw,eval_assd_def,eval_assd_raw,eval_tre_def,eval_tre_raw\
        ,eval_det=metrics
    dsc_trans,dsc_raw,hd95_def,hd95_raw,assd_def,assd_raw,tre_def,tre_raw\
        ,jac_det=metric

    eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

    eval_dsc_def.update(dsc_trans.item(), x.size(0))
    eval_dsc_raw.update(dsc_raw.item(), x.size(0))
    eval_hd95_def.update(hd95_def, x.size(0))
    eval_hd95_raw.update(hd95_raw, x.size(0))
    eval_assd_def.update(assd_def, x.size(0))
    eval_assd_raw.update(assd_raw, x.size(0))
    eval_tre_def.update(tre_def, x.size(0))
    eval_tre_raw.update(tre_raw, x.size(0))

def print_metric(metrics):

    eval_dsc_def,eval_dsc_raw,eval_hd95_def,eval_hd95_raw,eval_assd_def,eval_assd_raw,eval_tre_def,eval_tre_raw\
        ,eval_det=metrics


    print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                eval_dsc_def.std,
                                                                                eval_dsc_raw.avg,
                                                                                eval_dsc_raw.std))
    print('Deformed HD95: {:.3f} +- {:.3f}, Affine HD95: {:.3f} +- {:.3f}'.format(eval_hd95_def.avg,
                                                                                  eval_hd95_def.std,
                                                                                  eval_hd95_raw.avg,
                                                                                  eval_hd95_raw.std))
    print('Deformed ASSD: {:.3f} +- {:.3f}, Affine ASSD: {:.3f} +- {:.3f}'.format(eval_assd_def.avg,
                                                                                  eval_assd_def.std,
                                                                                  eval_assd_raw.avg,
                                                                                  eval_assd_raw.std))
    print('Deformed TRE: {:.3f} +- {:.3f}, Affine TRE: {:.3f} +- {:.3f}'.format(eval_tre_def.avg,
                                                                                eval_tre_def.std,
                                                                                eval_tre_raw.avg,
                                                                                eval_tre_raw.std))

    print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))