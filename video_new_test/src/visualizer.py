#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频物体显示器
使用固定坐标文件标识并显示视频中的物体
"""

import os
# 设置OpenCV后端以避免Qt显示问题
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re


class VideoObjectDisplay:
    """视频物体显示器"""
    
    def __init__(self, video_path, detection_info_path):
        """
        初始化显示器
        
        Args:
            video_path: 视频文件路径
            detection_info_path: 检测信息文件路径
        """
        self.video_path = video_path
        self.detection_info_path = detection_info_path
        
        # 加载固定物体坐标信息
        self.objects = self.load_detection_info()
        
        # 加载中文字体（如果可用）
        self.load_chinese_font()
        
        # 颜色设置
        self.colors = {
            'bbox': (0, 255, 0),        # 绿色边框
            'center': (0, 0, 255),      # 红色中心点
            'text': (255, 255, 255),    # 白色文字
            'text_bg': (0, 0, 0)        # 黑色文字背景
        }
    
    def load_chinese_font(self):
        """加载中文字体"""
        try:
            # 尝试加载项目中的中文字体
            # 使用相对于脚本的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            font_path = os.path.join(project_root, "font/SourceHan/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf")
            
            if os.path.exists(font_path):
                self.chinese_font = ImageFont.truetype(font_path, 24)
                self.chinese_font_small = ImageFont.truetype(font_path, 16)
                print(f"✓ 中文字体加载成功: {font_path}")
            else:
                print(f"警告: 中文字体文件不存在: {font_path}")
                self.chinese_font = ImageFont.load_default()
                self.chinese_font_small = ImageFont.load_default()
        except Exception as e:
            print(f"警告: 中文字体加载失败 {e}, 使用默认字体")
            self.chinese_font = ImageFont.load_default()
            self.chinese_font_small = ImageFont.load_default()
    
    def draw_chinese_text(self, image, text, position, font, color=(255, 255, 255)):
        """
        在图像上绘制中文文本
        Args:
            image: OpenCV图像 (BGR格式)
            text: 要绘制的文本
            position: 文本位置 (x, y)
            font: PIL字体对象
            color: 文本颜色 (B, G, R)
        Returns:
            绘制了文本的图像
        """
        # 将OpenCV图像转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 绘制文本 (PIL使用RGB格式)
        rgb_color = (color[2], color[1], color[0])  # BGR转RGB
        draw.text(position, text, font=font, fill=rgb_color)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def load_detection_info(self):
        """
        加载检测信息文件
        
        Returns:
            list: 物体信息列表
        """
        objects = []
        
        try:
            with open(self.detection_info_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用正则表达式解析目标信息
            targets = re.findall(r'目标 (\d+):(.*?)(?=目标 \d+:|$)', content, re.DOTALL)
            
            for target_num, target_content in targets:
                obj_info = {}
                
                # 解析类别信息
                class_match = re.search(r'类别: (.+?) \(ID: (\d+)\)', target_content)
                if class_match:
                    obj_info['class_name'] = class_match.group(1)
                    obj_info['class_id'] = int(class_match.group(2))
                
                # 解析置信度
                conf_match = re.search(r'置信度: ([\d.]+)', target_content)
                if conf_match:
                    obj_info['confidence'] = float(conf_match.group(1))
                
                # 解析边框坐标
                bbox_coords = {}
                coord_patterns = [
                    (r'左上角: \(([\d.]+), ([\d.]+)\)', 'x1', 'y1'),
                    (r'右下角: \(([\d.]+), ([\d.]+)\)', 'x2', 'y2')
                ]
                
                for pattern, x_key, y_key in coord_patterns:
                    coord_match = re.search(pattern, target_content)
                    if coord_match:
                        bbox_coords[x_key] = float(coord_match.group(1))
                        bbox_coords[y_key] = float(coord_match.group(2))
                
                # 解析中心点坐标
                center_match = re.search(r'中心点坐标: \(([\d.]+), ([\d.]+)\)', target_content)
                if center_match:
                    obj_info['center_x'] = float(center_match.group(1))
                    obj_info['center_y'] = float(center_match.group(2))
                
                # 如果有完整的边界框信息，设置bbox
                if len(bbox_coords) == 4:
                    obj_info['bbox'] = [bbox_coords['x1'], bbox_coords['y1'], 
                                       bbox_coords['x2'], bbox_coords['y2']]
                
                # 添加目标编号
                obj_info['target_num'] = int(target_num)
                
                if 'class_name' in obj_info and 'center_x' in obj_info:
                    objects.append(obj_info)
            
            print(f"✓ 加载检测信息: 共 {len(objects)} 个物体")
            for obj in objects:
                print(f"  - 目标{obj['target_num']}: {obj['class_name']} (ID: {obj['class_id']}) "
                      f"中心点({obj['center_x']:.1f}, {obj['center_y']:.1f}) "
                      f"置信度: {obj['confidence']:.4f}")
            
        except Exception as e:
            print(f"❌ 加载检测信息失败: {e}")
        
        return objects
    
    def draw_objects(self, frame):
        """
        在帧上绘制物体信息
        
        Args:
            frame: 输入图像帧
            
        Returns:
            绘制了物体信息的图像帧
        """
        annotated_frame = frame.copy()
        
        for obj in self.objects:
            # 绘制边界框
            if 'bbox' in obj:
                x1, y1, x2, y2 = obj['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 绘制矩形边框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.colors['bbox'], 2)
                
                # 在边框左上角绘制类别标签
                label = f"{obj['target_num']}: {obj['class_name']}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 绘制标签背景
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 10, y1), 
                            self.colors['text_bg'], -1)
                
                # 使用中文字体绘制标签
                annotated_frame = self.draw_chinese_text(
                    annotated_frame, label, (x1 + 5, y1 - label_size[1] - 5), 
                    self.chinese_font_small, self.colors['text']
                )
            
            # 绘制中心点
            center_x = int(obj['center_x'])
            center_y = int(obj['center_y'])
            
            # 绘制中心点（红色圆圈）
            cv2.circle(annotated_frame, (center_x, center_y), 8, self.colors['center'], -1)
            cv2.circle(annotated_frame, (center_x, center_y), 12, self.colors['center'], 2)
            
            # 在中心点旁边显示置信度
            conf_text = f"{obj['confidence']:.3f}"
            cv2.putText(annotated_frame, conf_text, 
                       (center_x + 15, center_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return annotated_frame
    
    def process_video(self, output_path=None, show_realtime=True):
        """
        处理视频文件
        
        Args:
            output_path: 输出视频路径（可选）
            show_realtime: 是否实时显示视频
        """
        # 检测是否有显示环境，如果没有则自动关闭实时显示
        try:
            import os
            if 'DISPLAY' not in os.environ and show_realtime:
                print("⚠️  检测到无显示环境，自动关闭实时显示模式")
                show_realtime = False
        except:
            pass
        
        # 打开视频文件
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 验证帧率
        if fps <= 0 or fps > 240:
            fps = 25  # 默认帧率
            print(f"⚠️  使用默认帧率: {fps} FPS")
        
        print(f"视频信息: {width}x{height}, {fps} FPS, 共 {total_frames} 帧")
        print(f"检测到 {len(self.objects)} 个固定物体")
        
        # 创建视频写入器（如果需要保存）
        out_video = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"将保存标注视频到: {output_path}")
        
        frame_count = 0
        
        print("开始处理视频...")
        if show_realtime:
            print("按 'q' 键退出，按 'p' 键暂停/继续")
        else:
            print("无GUI模式：将处理所有帧并保存到输出文件")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # 显示进度
                    if frame_count % 30 == 0 or frame_count == 1:  # 每30帧显示一次进度
                        progress = (frame_count / total_frames) * 100
                        print(f"处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")
                    
                    # 绘制物体信息
                    annotated_frame = self.draw_objects(frame)
                    
                    # 在帧上显示帧数信息
                    frame_info = f"Frame: {frame_count}/{total_frames} | Objects: {len(self.objects)}"
                    cv2.putText(annotated_frame, frame_info, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 保存到输出视频
                    if out_video:
                        out_video.write(annotated_frame)
                    
                    # 实时显示（仅在有显示环境时）
                    if show_realtime:
                        try:
                            cv2.imshow('Video Object Display', annotated_frame)
                            # 处理键盘输入
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                print("用户按下 'q' 键，退出处理")
                                break
                            elif key == ord('p'):
                                paused = not paused
                                print(f"视频 {'暂停' if paused else '继续'}")
                        except cv2.error as e:
                            print(f"显示错误，切换到无GUI模式: {e}")
                            show_realtime = False
                            cv2.destroyAllWindows()
                            
        except KeyboardInterrupt:
            print("\n用户中断处理")
        
        # 释放资源
        cap.release()
        if out_video:
            out_video.release()
        if show_realtime:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        print(f"视频处理完成！共处理 {frame_count} 帧")
        if output_path:
            print(f"标注视频已保存到: {output_path}")


def main():
    """主函数"""
    # 设置路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    video_path = os.path.join(project_root, 'datasets/双人操作/双人操作.mp4')
    detection_info_path = os.path.join(project_root, 'datasets/detection_info_254.txt')
    output_path = os.path.join(project_root, 'results/annotated_video_with_fixed_os.mp4')
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    if not os.path.exists(detection_info_path):
        print(f"❌ 检测信息文件不存在: {detection_info_path}")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        print("🚀 启动视频物体显示器")
        print(f"📹 输入视频: {video_path}")
        print(f"📋 检测信息: {detection_info_path}")
        print(f"💾 输出视频: {output_path}")
        
        # 创建显示器实例
        display = VideoObjectDisplay(video_path, detection_info_path)
        
        # 处理视频 - 在无显示环境中自动关闭实时显示
        display.process_video(output_path, show_realtime=False)  # 默认关闭实时显示
        
        print("✅ 处理完成！")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
