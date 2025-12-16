import numpy as np
import cv2
import open3d as o3d

# 自定义工具函数（根据可视化代码需求实现）
def get_color(color_name):
    """根据颜色名称返回对应的RGB颜色（0-1范围）"""
    color_map = {
        "custom_yellow": [1.0, 0.84, 0.0],  # 金色/黄色
        "custom_blue": [0.0, 0.4, 0.8]      # 深蓝色
    }
    return color_map.get(color_name, [0.5, 0.5, 0.5])  # 默认灰色

def make_open3d_point_cloud(points, colors=None, normals=None):
    """创建Open3D点云对象"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def draw_geometries(*geometries):
    """可视化多个几何对象"""
    o3d.visualization.draw_geometries(list(geometries), window_name="Point Cloud Visualization")

def depth_to_pointcloud(depth_img, fx, fy, cx, cy, scale=1000.0):
    """
    将深度图转换为点云（相机坐标系）
    
    参数:
        depth_img: 深度图像（numpy数组，uint16或float32）
        fx, fy: 相机焦距（像素单位）
        cx, cy: 相机主点（像素单位）
        scale: 深度缩放因子（默认1000，因为Kinect等设备的深度值单位是毫米）
    
    返回:
        points: 点云数组 (H*W, 3)，每行是(x, y, z)
    """
    # 获取图像尺寸
    H, W = depth_img.shape
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    
    # 转换为相机坐标系下的3D点
    # 公式：x = (u - cx) * z / fx
    #      y = (v - cy) * z / fy
    #      z = depth / scale
    
    # 处理无效深度值（0或负数）
    valid_mask = depth_img > 0
    z = depth_img[valid_mask] / scale
    
    # 计算x和y坐标
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy
    
    # 组合为点云数组
    points = np.column_stack((x, y, z))
    
    return points



def main():
    ############### 配置参数 #################
    # 1. 深度图路径（请替换为你的实际文件路径）
    # ref_depth_path = "F:/PaperWriting/PesudoDepth/experiments/draw_251113/ycbv/56_53/depth/000056_000053_dp1.png"  # 参考深度图
    # src_depth_path = "F:/PaperWriting/PesudoDepth/experiments/draw_251113/ycbv/56_53/depth/000056_000053_rdp1.png"    # 源深度图

    ref_depth_path = "F:/PaperWriting/PesudoDepth/experiments/ycbv_depth/depth_DAS/000048/000036.png"  # 参考深度图
    src_depth_path = "F:/PaperWriting/PesudoDepth/experiments/ycbv_depth/depth_MoGe-L/000048/depth_MoGe-L/000036.png"    # 源深度图
    # src_depth_path = "/home/data1/wangqingyuan/PseudoFlow/data/ycbv/test_bop19/000048/depth/000001.png"    # 源深度图
    
    # [1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0]
    # 2. 相机内参（请根据你的相机实际参数修改！）
    # 格式：fx, fy (焦距), cx, cy (主点坐标)
    fx = 1066.778    # 示例：Kinect v1内参
    fy = 1067.487
    cx = 312.9869
    cy = 241.3109
    
    # 3. 深度缩放因子（根据深度图的单位调整）
    # - 如果深度值单位是毫米 → scale=1000.0
    # - 如果深度值单位是米 → scale=1.0
    # - 如果是其他单位（如厘米）→ scale=100.0
    depth_scale1 = 1000.0 # 1000.0
    # depth_scale2 = 96.5 # 1000.0  ycbv 000048
    # depth_scale2 = 125.0 # 1000.0  ycbv 000050
    depth_scale2 = 85.0 # 1000.0  ycbv 000056

    ##########################################
    
    ############### 读取深度图 #################
    # 读取16位深度图（PNG格式通常存储为uint16）
    ref_depth = cv2.imread(ref_depth_path, cv2.IMREAD_UNCHANGED)
    src_depth = cv2.imread(src_depth_path, cv2.IMREAD_UNCHANGED)
    
    # 检查读取是否成功
    if ref_depth is None:
        raise FileNotFoundError(f"无法读取参考深度图: {ref_depth_path}")
    if src_depth is None:
        raise FileNotFoundError(f"无法读取源深度图: {src_depth_path}")
    
    # 确保深度图是单通道的
    if len(ref_depth.shape) > 2:
        ref_depth = ref_depth[..., 0]
    if len(src_depth.shape) > 2:
        src_depth = src_depth[..., 0]
    
    ############### 转换为点云 #################
    print("正在将深度图转换为点云...")
    ref_points = depth_to_pointcloud(ref_depth, fx, fy, cx, cy, 1)
    src_points = depth_to_pointcloud(src_depth, fx, fy, cx, cy, depth_scale1)
    
    print(f"参考点云数量: {ref_points.shape[0]}")
    print(f"源点云数量: {src_points.shape[0]}")
    
    ############### 数据整理 #################
    data_dict = {
        "ref_points": ref_points,
        "src_points": src_points
    }
    
    ############### visualization #################
    ref_points_reshaped = data_dict["ref_points"].reshape(-1, 3)
    src_points_reshaped = data_dict["src_points"].reshape(-1, 3)
    
    ref_pcd = make_open3d_point_cloud(ref_points_reshaped)
    ref_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )  # 优化法线估计参数
    ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    
    src_pcd = make_open3d_point_cloud(src_points_reshaped)
    src_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    src_pcd.paint_uniform_color(get_color("custom_blue"))
    
    draw_geometries(ref_pcd, ref_pcd)
    ###############################################

if __name__ == "__main__":
    main()