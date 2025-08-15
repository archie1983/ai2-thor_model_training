#!/usr/bin/env python3
"""
使用 thor_topdown_img 函数生成俯视图图片的示例脚本
参考 thor_topdown_img 函数的实现原理
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from thortils.controller import launch_controller
from thortils.vision import thor_topdown_img

class TopdownImageGenerator:
    """俯视图图片生成器类"""
    
    def __init__(self, save_dir="generated_images"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def get_topdown_image(self, scene_name, controller_config=None):
        """
        获取指定场景的俯视图图片
        
        Args:
            scene_name (str): 场景名称
            controller_config (dict): 控制器配置
            
        Returns:
            PIL.Image: 俯视图图片
        """
        if controller_config is None:
            controller_config = {'scene': scene_name, 'agentMode': 'default'}
        
        try:
            print(f"正在生成 {scene_name} 的俯视图...")
            
            # 启动控制器
            controller = launch_controller(controller_config)
            
            # 使用 thor_topdown_img 函数获取俯视图
            # 这个函数内部会：
            # 1. 切换到地图视图 (ToggleMapView)
            # 2. 等待渲染完成 (Pass)
            # 3. 获取图片
            # 4. 切换回正常视图
            img_array = thor_topdown_img(controller)
            
            # 确保图片格式正确
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 3:
                    # BGR 转 RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                elif img_array.shape[2] == 4:
                    # BGRA 转 RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
            
            controller.stop()
            
            return Image.fromarray(img_array)
            
        except Exception as e:
            print(f"生成 {scene_name} 俯视图时出错: {e}")
            # 返回错误占位图片
            return Image.new("RGB", (400, 400), (255, 0, 0))
    
    def generate_single_scene(self, scene_name, output_name=None):
        """生成单个场景的俯视图"""
        if output_name is None:
            output_name = f"{scene_name}_topdown.png"
        
        img = self.get_topdown_image(scene_name)
        output_path = os.path.join(self.save_dir, output_name)
        img.save(output_path, dpi=300, quality=95)
        print(f"图片已保存: {output_path}")
        return img
    
    def generate_scene_grid(self, scene_names, rows, cols, output_name="scene_grid.png"):
        """生成场景网格图"""
        if len(scene_names) > rows * cols:
            print(f"警告: 场景数量 ({len(scene_names)}) 超过网格大小 ({rows}x{cols})")
            scene_names = scene_names[:rows*cols]
        
        # 获取所有场景的图片
        images = []
        for scene in scene_names:
            img = self.get_topdown_image(scene)
            images.append(img)
        
        # 统一尺寸
        max_w = max(img.width for img in images)
        max_h = max(img.height for img in images)
        
        # 调整所有图片尺寸
        resized_images = []
        for img in images:
            if img.width != max_w or img.height != max_h:
                img = img.resize((max_w, max_h), Image.Resampling.LANCZOS)
            resized_images.append(img)
        
        # 补齐空白图片
        while len(resized_images) < rows * cols:
            blank_img = Image.new("RGB", (max_w, max_h), (240, 240, 240))
            resized_images.append(blank_img)
        
        # 创建网格
        grid_rows = []
        for i in range(rows):
            row_images = resized_images[i*cols:(i+1)*cols]
            row_concat = np.concatenate([np.array(img) for img in row_images], axis=1)
            grid_rows.append(row_concat)
        
        # 垂直拼接
        final_grid = np.concatenate(grid_rows, axis=0)
        final_img = Image.fromarray(final_grid)
        
        # 保存
        output_path = os.path.join(self.save_dir, output_name)
        final_img.save(output_path, dpi=300, quality=95)
        print(f"网格图已保存: {output_path}")
        
        return final_img
    
    def generate_comparison(self, scene_names, output_name="scene_comparison.png"):
        """生成场景对比图"""
        if len(scene_names) < 2:
            print("至少需要2个场景进行对比")
            return None
        
        images = []
        for scene in scene_names:
            img = self.get_topdown_image(scene)
            images.append(img)
        
        # 统一尺寸
        max_w = max(img.width for img in images)
        max_h = max(img.height for img in images)
        
        # 调整尺寸
        resized_images = []
        for img in images:
            if img.width != max_w or img.height != max_h:
                img = img.resize((max_w, max_h), Image.Resampling.LANCZOS)
            resized_images.append(img)
        
        # 水平拼接
        final_img = np.concatenate([np.array(img) for img in resized_images], axis=1)
        final_img = Image.fromarray(final_img)
        
        # 保存
        output_path = os.path.join(self.save_dir, output_name)
        final_img.save(output_path, dpi=300, quality=95)
        print(f"对比图已保存: {output_path}")
        
        return final_img

def main():
    """主函数示例"""
    generator = TopdownImageGenerator()
    
    # 示例1: 生成单个场景俯视图
    print("=== 生成单个场景俯视图 ===")
    single_img = generator.generate_single_scene("FloorPlan1", "floorplan1_topdown.png")
    
    # 示例2: 生成场景网格图
    print("\n=== 生成场景网格图 ===")
    scenes = ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6"]
    grid_img = generator.generate_scene_grid(scenes, 2, 3, "floorplan_grid.png")
    
    # 示例3: 生成场景对比图
    print("\n=== 生成场景对比图 ===")
    comparison_img = generator.generate_comparison(scenes[:4], "floorplan_comparison.png")
    
    # 示例4: 使用ProcTHOR场景
    print("\n=== 生成ProcTHOR场景俯视图 ===")
    procthor_scenes = ["ProcTHOR-00001", "ProcTHOR-00002", "ProcTHOR-00003"]
    try:
        procthor_grid = generator.generate_scene_grid(procthor_scenes, 1, 3, "procthor_grid.png")
    except Exception as e:
        print(f"ProcTHOR场景生成失败: {e}")
    
    print(f"\n所有图片已保存到: {generator.save_dir}")

if __name__ == "__main__":
    main() 