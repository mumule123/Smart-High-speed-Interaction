#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备操作检测系统 - 简化版本
使用固定坐标而不进行设备识别，结合人体姿态识别分析操作行为
"""

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import argparse
from collections import defaultdict
import json  # 仅用于保存分析报告
from datetime import datetime
import math
from PIL import Image, ImageDraw, ImageFont

# 导入配置管理器
from config_manager import ConfigManager


class OperationDetector:
    def __init__(self, project_root, config_file_path):
        """
        初始化操作检测器
        
        Args:
            project_root: 项目根目录路径
            config_file_path: 配置文件路径
        """
        self.project_root = project_root
        
        # 加载配置文件（使用配置管理器）
        config_manager = ConfigManager()
        self.config = config_manager.load_config(config_file_path)
        
        # 设备类别映射（从配置中获取）
        self.all_classes = self.config['all_classes']
        
        # 操作判断阈值（像素距离，从配置文件获取）
        self.operation_threshold = self.config['thresholds']['operation_distance_threshold']
        
        # 加载模型（只加载姿态识别模型，不加载目标检测模型）
        self.load_models()
        
        # 操作记录
        self.operation_records = defaultdict(list)
        
        # 加载中文字体
        self.load_chinese_font()
    
    def get_target_devices_from_detection_info(self):
        """
        从 detection_info_254.txt 文件中获取目标设备信息
        包括设备ID、坐标等信息，特别处理高压隔离开关的4个设备
        """
        detection_info_file = os.path.join(self.project_root, self.config['file_paths']['fixed_coords_file'])
        target_devices = []
        
        try:
            with open(detection_info_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    if line.startswith('目标'):
                        device_info = {}
                        i += 1
                        
                        # 解析类别信息
                        if i < len(lines) and '类别:' in lines[i]:
                            class_line = lines[i].strip()
                            if '(' in class_line and 'ID:' in class_line:
                                class_name = class_line.split('类别:')[1].split('(')[0].strip()
                                class_id_str = class_line.split('ID:')[1].split(')')[0].strip()
                                device_info['class_name'] = class_name
                                device_info['class_id'] = int(class_id_str)
                            i += 1
                        
                        # 解析置信度
                        if i < len(lines) and '置信度:' in lines[i]:
                            conf_line = lines[i].strip()
                            confidence = float(conf_line.split('置信度:')[1].strip())
                            device_info['confidence'] = confidence
                            i += 1
                        
                        # 跳过边框坐标标题行
                        if i < len(lines) and '边框坐标:' in lines[i]:
                            i += 1
                        
                        # 解析坐标信息 
                        bbox_coords = {}
                        for _ in range(4):
                            if i < len(lines):
                                coord_line = lines[i].strip()
                                if '(' in coord_line and ')' in coord_line:
                                    coord_str = coord_line.split('(')[1].split(')')[0]
                                    x, y = map(float, coord_str.split(','))
                                    if '左上角' in coord_line:
                                        bbox_coords['x1'] = x
                                        bbox_coords['y1'] = y
                                    elif '右下角' in coord_line:
                                        bbox_coords['x2'] = x
                                        bbox_coords['y2'] = y
                                i += 1
                        
                        # 解析中心点坐标
                        if i < len(lines) and '中心点坐标:' in lines[i]:
                            center_line = lines[i].strip()
                            if '(' in center_line and ')' in center_line:
                                coord_str = center_line.split('(')[1].split(')')[0]
                                center_x, center_y = map(float, coord_str.split(','))
                                device_info['center_x'] = center_x
                                device_info['center_y'] = center_y
                            i += 1
                        
                        # 如果解析到了有效信息，添加到列表中
                        if 'class_id' in device_info and 'center_x' in device_info:
                            if 'x1' in bbox_coords and 'x2' in bbox_coords:
                                device_info['bbox'] = [bbox_coords['x1'], bbox_coords['y1'], 
                                                     bbox_coords['x2'], bbox_coords['y2']]
                            else:
                                estimation_size = self.config['device_detection']['missing_device_estimation_size']
                                device_info['bbox'] = [device_info['center_x'] - estimation_size, device_info['center_y'] - estimation_size,
                                                     device_info['center_x'] + estimation_size, device_info['center_y'] + estimation_size]
                            
                            target_devices.append(device_info)
                    else:
                        i += 1
            
            print(f"✓ 从检测信息文件加载目标设备: 共 {len(target_devices)} 个设备")
            
            # 统计每种设备类型的数量，特别标记高压隔离开关
            device_count = {}
            for device in target_devices:
                class_name = device['class_name']
                class_id = device['class_id']
                key = f"{class_name} (ID: {class_id})"
                device_count[key] = device_count.get(key, 0) + 1
            
            for device_type, count in device_count.items():
                if count > 1:
                    print(f"  - {device_type}: {count} 个设备")
                else:
                    print(f"  - {device_type}: {count} 个设备")
            
        except Exception as e:
            print(f"警告: 从检测信息文件加载目标设备失败 {e}")
            
        return target_devices
    
    def load_chinese_font(self):
        """加载中文字体"""
        try:
            # 中文字体路径
            font_path = os.path.join(self.project_root, self.config['font_settings']['chinese_font_path'])
            if os.path.exists(font_path):
                font_size_normal = self.config['font_settings']['font_size_normal']
                font_size_small = self.config['font_settings']['font_size_small']
                self.chinese_font = ImageFont.truetype(font_path, font_size_normal)
                self.chinese_font_small = ImageFont.truetype(font_path, font_size_small)
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
        
    def load_models(self):
        """只加载姿态识别模型，不加载目标检测模型"""
        try:
            # 加载姿态识别模型
            pose_model_path = os.path.join(self.project_root, self.config['models']['pose_model_path'])
            self.pose_model = YOLO(pose_model_path)
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def get_hand_keypoints(self, keypoints):
        """
        从YOLO pose检测结果中提取手部关键点
        返回左右手腕和手肘的坐标
        """
        # YOLO pose关键点索引（COCO格式）- 从配置中获取
        keypoint_indices = self.config['keypoint_indices']
        
        hand_points = {}
        
        for point_name, index in keypoint_indices.items():
            if index < len(keypoints):
                x, y, confidence = keypoints[index]
                # 使用配置中的置信度阈值
                keypoint_threshold = self.config['thresholds']['keypoint_confidence_threshold']
                if confidence > keypoint_threshold:
                    hand_points[point_name] = {'x': float(x), 'y': float(y), 'confidence': float(confidence)}
                else:
                    hand_points[point_name] = {'x': None, 'y': None, 'confidence': float(confidence)}
            else:
                hand_points[point_name] = {'x': None, 'y': None, 'confidence': 0.0}
        
        return hand_points
    
    def calculate_weighted_hand_center(self, hand_points):
        """
        使用加权公式计算手部中心点
        公式: 中心点位置 = 0.35*left_wrist + 0.35*right_wrist + 0.15*right_elbow + 0.15*left_elbow
        """
        left_wrist = hand_points['left_wrist']
        right_wrist = hand_points['right_wrist']
        left_elbow = hand_points['left_elbow']
        right_elbow = hand_points['right_elbow']
        
        # 检查是否有足够的有效点来计算中心点
        valid_points = []
        weights = []
        
        # 从配置文件获取权重
        weights_config = self.config['hand_center_weights']
        
        if left_wrist['x'] is not None and left_wrist['y'] is not None:
            valid_points.append((left_wrist['x'], left_wrist['y']))
            weights.append(weights_config['left_wrist_weight'])
            
        if right_wrist['x'] is not None and right_wrist['y'] is not None:
            valid_points.append((right_wrist['x'], right_wrist['y']))
            weights.append(weights_config['right_wrist_weight'])
            
        if left_elbow['x'] is not None and left_elbow['y'] is not None:
            valid_points.append((left_elbow['x'], left_elbow['y']))
            weights.append(weights_config['left_elbow_weight'])
            
        if right_elbow['x'] is not None and right_elbow['y'] is not None:
            valid_points.append((right_elbow['x'], right_elbow['y']))
            weights.append(weights_config['right_elbow_weight'])
        
        # 如果没有有效点，返回None
        if len(valid_points) == 0:
            return {'x': None, 'y': None}
        
        # 根据有效点重新计算权重（确保权重和为1）
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 计算加权中心点
        center_x = sum(p[0] * w for p, w in zip(valid_points, normalized_weights))
        center_y = sum(p[1] * w for p, w in zip(valid_points, normalized_weights))
        
        return {'x': float(center_x), 'y': float(center_y)}
    
    
    
    def calculate_center_point(self, points):
        """计算多个点的中心位置，支持列表或字典输入"""
        # 如果是字典（如 hand_points），取其 values
        if isinstance(points, dict):
            points = list(points.values())
        valid_points = []
        for point in points:
            if isinstance(point, dict) and point.get('x') is not None and point.get('y') is not None:
                valid_points.append((point['x'], point['y']))
        if len(valid_points) == 0:
            return {'x': None, 'y': None}
        center_x = sum(p[0] for p in valid_points) / len(valid_points)
        center_y = sum(p[1] for p in valid_points) / len(valid_points)
        return {'x': float(center_x), 'y': float(center_y)}
    
    
    
    
    def calculate_distance(self, point1, point2):
        """计算两点之间的欧几里得距离"""
        if (point1['x'] is None or point1['y'] is None or 
            point2['x'] is None or point2['y'] is None):
            return float('inf')
        
        dx = point1['x'] - point2['x']
        dy = point1['y'] - point2['y']
        return math.sqrt(dx * dx + dy * dy)
    
    def detect_objects(self, frame, frame_number=0, timestamp=0.0):
        """
        直接使用固定坐标作为设备位置，不进行实际的YOLO检测
        
        Args:
            frame: 输入图像帧
            frame_number: 当前帧号
            timestamp: 当前时间戳(秒)（用于缓存管理）
            
        Returns:
            list: 检测到的设备列表
        """
        # 直接从固定坐标文件中获取所有设备信息
        target_device_info = self.get_target_devices_from_detection_info()
        detected_objects = []
        
        # 使用固定坐标置信度
        fixed_confidence = self.config['thresholds']['fixed_coordinate_confidence']
        
        print(f"� 帧 {frame_number}: 使用固定坐标加载所有设备")
        
        # 直接将所有固定坐标设备添加到检测结果中
        for device in target_device_info:
            detected_objects.append({
                'class_id': device['class_id'],
                'class_name': device['class_name'],
                'center': {'x': device['center_x'], 'y': device['center_y']},
                'bbox': device['bbox'],
                'confidence': fixed_confidence,
                'source': 'fixed_coordinates'  # 标记来源为固定坐标
            })
        
        print(f"� 帧 {frame_number}: 固定坐标模式加载了 {len(detected_objects)} 个设备")
        
        return detected_objects
    
    def detect_poses(self, frame):
        """使用姿态识别模型检测人体姿态"""
        results = self.pose_model(frame)
        detected_persons = []
        
        for r in results:
            if r.keypoints is not None:
                for i, keypoints in enumerate(r.keypoints.data):
                    # 提取手部关键点
                    hand_points = self.get_hand_keypoints(keypoints)
                    
                    # 使用加权公式计算手部中心点
                    # 公式参数从配置文件获取  2钟选择，使用加权或者直接平均
                    hand_center = self.calculate_weighted_hand_center(hand_points)
                    
                    if hand_center['x'] is not None:  # 只有当能计算出手部中心时才添加
                        detected_persons.append({
                            'person_id': i,
                            'hand_keypoints': hand_points,
                            'hand_center': hand_center
                        })
        
        return detected_persons
    
    def analyze_operations(self, persons, objects, frame_number, timestamp):
        """分析人员操作行为"""
        operations = []
        
        for person in persons:
            if person['hand_center']['x'] is None:
                continue
                
            min_distance = float('inf')
            closest_object = None
            
            # 找到离手部中心最近的物体
            for obj in objects:
                distance = self.calculate_distance(person['hand_center'], obj['center'])
                if distance < min_distance:
                    min_distance = distance
                    closest_object = obj
            
            # 如果距离小于阈值，认为正在操作
            if closest_object and min_distance <= self.operation_threshold:
                operation = {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'person_id': person['person_id'],
                    'device_class_id': closest_object['class_id'],
                    'device_name': closest_object['class_name'],
                    'distance': min_distance,
                    'hand_center': person['hand_center'],
                    'device_center': closest_object['center'],
                    'device_confidence': closest_object['confidence']
                }
                operations.append(operation)
                
                # 记录操作历史
                self.operation_records[closest_object['class_id']].append(operation)
        
        return operations
    
    def draw_annotations(self, frame, objects, persons, operations):
        """在帧上绘制检测结果和操作分析"""
        annotated_frame = frame.copy()
        
        # 绘制检测到的物体
        for obj in objects:
            center_x, center_y = int(obj['center']['x']), int(obj['center']['y'])
            
            # 设备中心点颜色和半径从配置获取
            center_color = tuple(self.config['visualization']['device_center_color'])
            center_radius = self.config['visualization']['device_center_radius']
            
            # 绘制中心点
            cv2.circle(annotated_frame, (center_x, center_y), center_radius, center_color, -1)
        
        # 绘制人体手部关键点
        hand_colors = self.config['hand_colors']
        keypoint_radius = self.config['visualization']['keypoint_radius']
        
        for person in persons:
            # 绘制手部关键点
            for point_name, point_data in person['hand_keypoints'].items():
                if point_data['x'] is not None and point_data['y'] is not None:
                    x, y = int(point_data['x']), int(point_data['y'])
                    color = tuple(hand_colors.get(point_name, [255, 255, 255]))
                    cv2.circle(annotated_frame, (x, y), keypoint_radius, color, -1)
                    cv2.putText(annotated_frame, point_name[:5], (x+8, y-8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # 绘制手部中心点
            if person['hand_center']['x'] is not None:
                center_x = int(person['hand_center']['x'])
                center_y = int(person['hand_center']['y'])
                hand_center_color = tuple(self.config['visualization']['hand_center_color'])
                hand_center_radius = self.config['visualization']['hand_center_radius']
                cv2.circle(annotated_frame, (center_x, center_y), hand_center_radius, hand_center_color, -1)  # 红色
                cv2.putText(annotated_frame, f'P{person["person_id"]}', (center_x+12, center_y-12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 绘制操作连线和标注
        for op in operations:
            hand_x = int(op['hand_center']['x'])
            hand_y = int(op['hand_center']['y'])
            device_x = int(op['device_center']['x'])
            device_y = int(op['device_center']['y'])
            
            # 绘制连线
            line_color = tuple(self.config['visualization']['operation_line_color'])
            line_thickness = self.config['visualization']['operation_line_thickness']
            cv2.line(annotated_frame, (hand_x, hand_y), (device_x, device_y), line_color, line_thickness)
            
            # 标注操作信息（使用中文字体）
            mid_x = (hand_x + device_x) // 2
            mid_y = (hand_y + device_y) // 2
            operation_text = f"操作中: {op['device_name']}"
            distance_text = f"距离: {op['distance']:.1f}px"
            
            # 使用中文字体绘制操作信息
            text_color = tuple(self.config['visualization']['operation_text_color'])
            annotated_frame = self.draw_chinese_text(
                annotated_frame, operation_text, (mid_x, mid_y-25), self.chinese_font_small, text_color
            )
            annotated_frame = self.draw_chinese_text(
                annotated_frame, distance_text, (mid_x, mid_y+5), self.chinese_font_small, text_color
            )
        
        return annotated_frame
    
    def process_video(self, video_path, output_dir):
        """处理视频文件"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 验证帧率有效性，使用配置文件作为回退
        if fps <= 0 or fps > 240:
            original_fps = fps
            fps = self.config['video_processing']['fps_assumption']
            print(f"⚠️  视频帧率无效 ({original_fps} FPS)，使用配置中的回退帧率: {fps} FPS")
        
        # 保存实际使用的帧率供后续计算使用
        self.actual_fps = fps
        
        print(f"视频信息: {width}x{height}, {fps} FPS, 共 {total_frames} 帧")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_dir, 'operation_analysis.mp4')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 数据记录
        all_frame_data = []
        frame_count = 0
        
        print("开始处理视频...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            if frame_count % 30 == 0:  # 每30帧显示进度
                print(f"处理进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # 检测物体和人体姿态
            detected_objects = self.detect_objects(frame, frame_count, timestamp)
            detected_persons = self.detect_poses(frame)
            
            # 分析操作行为
            operations = self.analyze_operations(detected_persons, detected_objects, frame_count, timestamp)
            
            # 绘制标注
            annotated_frame = self.draw_annotations(frame, detected_objects, detected_persons, operations)
            
            # 写入视频
            out_video.write(annotated_frame)
            
            # 记录帧数据
            frame_data = {
                'frame_number': frame_count,
                'timestamp': timestamp,
                'detected_objects': detected_objects,
                'detected_persons': detected_persons,
                'operations': operations
            }
            all_frame_data.append(frame_data)
        
        cap.release()
        out_video.release()
        
        print(f"视频处理完成！共处理 {frame_count} 帧")
        print(f"标注视频保存为: {output_video_path}")
        
        # 保存分析结果
        self.save_analysis_results(all_frame_data, output_dir)
        
        return all_frame_data
    
    def save_analysis_results(self, frame_data, output_dir):
        """保存分析结果到文件"""
        
        # 1. 计算每个设备的操作时间统计
        device_operation_stats = self.calculate_operation_time_stats()
        
        # 创建设备ID到设备名称的映射（从固定坐标文件中获取）
        target_devices = self.get_target_devices_from_detection_info()
        device_id_to_name = {}
        for device in target_devices:
            device_id_to_name[device['class_id']] = device['class_name']
        
        stats_df = pd.DataFrame([
            {
                'device_class_id': device_id,
                'device_name': device_id_to_name.get(device_id, f"Unknown_{device_id}"),
                'total_operation_frames': stats['total_frames'],
                'total_operation_time_seconds': stats['total_time'],
                'operation_episodes': stats['episodes'],
                'average_distance': stats['avg_distance']
            }
            for device_id, stats in device_operation_stats.items()
        ])
        
        stats_csv_path = os.path.join(output_dir, 'device_operation_stats.csv')
        stats_df.to_csv(stats_csv_path, index=False, encoding='utf-8')
        print(f"设备操作统计保存为: {stats_csv_path}")
        
        # 2. 保存完整的分析报告
        report = {
            'analysis_info': {
                'total_frames': len(frame_data),
                'total_duration_seconds': frame_data[-1]['timestamp'] if frame_data else 0,
                'operation_threshold_pixels': self.operation_threshold,
                'device_classes': device_id_to_name,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'device_operation_summary': device_operation_stats
        }
        
        report_path = os.path.join(output_dir, 'analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"分析报告保存为: {report_path}")
        
        # 4. 打印摘要信息
        print("\n" + "="*50)
        print("操作分析摘要")
        print("="*50)
        
        if device_operation_stats:
            print("\n各设备操作时间统计:")
            for device_id, stats in device_operation_stats.items():
                device_name = device_id_to_name.get(device_id, f"Unknown_{device_id}")
                print(f"  {device_name}:")
                print(f"    - 操作时间: {stats['total_time']:.2f} 秒")
                print(f"    - 操作帧数: {stats['total_frames']} 帧")
                print(f"    - 操作次数: {stats['episodes']} 次")
                print(f"    - 平均距离: {stats['avg_distance']:.2f} 像素")
        else:
            print("未检测到任何操作行为")
    
    def calculate_operation_time_stats(self):
        """计算每个设备的操作时间统计"""
        stats = {}
        
        for device_id, operations in self.operation_records.items():
            if not operations:
                continue
            
            # 按时间排序
            operations_sorted = sorted(operations, key=lambda x: x['timestamp'])
            
            # 计算连续操作的时间段
            episodes = []
            current_episode = [operations_sorted[0]]
            
            for i in range(1, len(operations_sorted)):
                # 如果两个操作之间的时间间隔小于2秒，认为是连续操作
                if operations_sorted[i]['timestamp'] - operations_sorted[i-1]['timestamp'] <= 2.0:
                    current_episode.append(operations_sorted[i])
                else:
                    episodes.append(current_episode)
                    current_episode = [operations_sorted[i]]
            
            episodes.append(current_episode)
            
            # 计算统计信息
            total_frames = len(operations)
            total_time = sum(len(episode) / self.actual_fps for episode in episodes)  # 使用实际帧率
            avg_distance = sum(op['distance'] for op in operations) / len(operations)
            
            stats[device_id] = {
                'total_frames': total_frames,
                'total_time': total_time,
                'episodes': len(episodes),
                'avg_distance': avg_distance
            }
        
        return stats


def main():
    """主函数"""
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 默认YAML配置文件路径
    default_config_path = os.path.join(project_root, 'config.yaml')
    
    parser = argparse.ArgumentParser(description='设备操作检测分析系统 V3')
    parser.add_argument('--config', type=str, default=default_config_path,
                       help='YAML配置文件路径 (默认: config.yaml)')
    parser.add_argument('--video', type=str, default=None,
                       help='输入视频文件路径 (可选，会覆盖配置文件中的设置)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出结果目录路径 (可选，会覆盖配置文件中的设置)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='操作判断距离阈值（像素） (可选，会覆盖配置文件中的设置)')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return
    
    try:
        # 初始化检测器，传入配置文件路径
        detector = OperationDetector(project_root, args.config)
        
        # 从配置文件获取默认参数，或使用命令行参数覆盖
        video_path = args.video if args.video else os.path.join(project_root, detector.config['file_paths']['default_video_path'])
        output_dir = args.output if args.output else os.path.join(project_root, detector.config['file_paths']['default_output_dir'])
        
        # 如果指定了命令行参数，则覆盖配置文件中的设置
        if args.threshold is not None:
            detector.operation_threshold = args.threshold
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return
        
        print("🚀 启动设备操作检测分析系统 V3 - 智能配置")
        print(f"📋 配置文件: {args.config}")
        print(f"📹 输入视频: {video_path}")
        print(f"📁 输出目录: {output_dir}")
        print(f"📏 距离阈值: {detector.operation_threshold} 像素")
        
        # 处理视频
        # 处理视频
        _ = detector.process_video(video_path, output_dir)
        
        print("✅ 分析完成！")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
