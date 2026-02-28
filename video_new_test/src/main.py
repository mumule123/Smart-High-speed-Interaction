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
import json  # 用于加载object.json和保存分析报告
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
        
        # 状态跟踪：记录每个人员对每个设备的操作状态和时间
        self.operation_states = defaultdict(lambda: defaultdict(dict))  # person_id -> device_id -> state_info
        
        # 时间阈值配置
        self.operation_start_time = 3.0  # 近距离超过3秒才算正在操作
        self.operation_end_time = 2.0    # 远离超过2秒才算结束操作
        
        # 加载设备多边形数据
        self.device_polygons = self.load_device_polygons_from_json()
        
        # 加载中文字体
        self.load_chinese_font()
    
    def load_device_polygons_from_json(self):
        """
        从object.json文件加载设备多边形数据
        返回设备名称到多边形信息的映射
        """
        json_file_path = os.path.join(self.project_root, 'datasets', 'object.json')
        device_polygons = {}
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 解析JSON数据结构
            if isinstance(data, list) and len(data) > 0:
                annotations = data[0].get('annotations', [])
                
                for annotation in annotations:
                    result = annotation.get('result', [])
                    
                    for item in result:
                        if item.get('type') == 'polygonlabels':
                            value = item.get('value', {})
                            points = value.get('points', [])
                            labels = value.get('polygonlabels', [])
                            
                            if points and labels:
                                device_name = labels[0]  # 取第一个标签作为设备名称
                                
                                # 坐标是百分比形式，需要保存原始数据
                                device_polygons[device_name] = {
                                    'points_percent': points,
                                    'original_width': item.get('original_width', 1920),
                                    'original_height': item.get('original_height', 1080),
                                    'id': item.get('id', '')
                                }
            
            print(f"✓ 从object.json加载了 {len(device_polygons)} 个设备多边形")
            for device_name in device_polygons.keys():
                print(f"  - {device_name}")
                
        except Exception as e:
            print(f"警告: 加载object.json失败 {e}")
            
        return device_polygons
    
    def convert_polygon_to_pixels(self, points_percent, img_width, img_height):
        """
        将百分比坐标转换为像素坐标
        Args:
            points_percent: 百分比坐标点列表 [[x1, y1], [x2, y2], ...]
            img_width: 图像宽度
            img_height: 图像高度
        Returns:
            像素坐标点列表
        """
        pixel_points = []
        for point in points_percent:
            x_pixel = int(point[0] * img_width / 100.0)
            y_pixel = int(point[1] * img_height / 100.0)
            pixel_points.append([x_pixel, y_pixel])
        return pixel_points
    
    def calculate_point_to_polygon_distance(self, point, polygon_points):
        """
        计算点到多边形边界的距离 - 使用OpenCV实现
        Args:
            point: 点坐标 {'x': x, 'y': y}
            polygon_points: 多边形像素坐标点列表
        Returns:
            float: 到多边形边界的距离（像素），如果在内部返回0
        """
        try:
            # 转换为numpy数组格式
            test_point = (point['x'], point['y'])
            polygon_array = np.array(polygon_points, dtype=np.int32)
            
            # 检查点到多边形的距离
            result = cv2.pointPolygonTest(polygon_array, test_point, True)
            
            # result > 0: 点在多边形内部，距离为正值
            # result = 0: 点在多边形边界上  
            # result < 0: 点在多边形外部，绝对值是到边界的距离
            
            if result >= 0:  # 在内部或边界上
                return 0.0  # 在内部或边界上，视为距离为0
            else:
                return abs(result)  # 在外部，返回到边界的距离
            
        except Exception as e:
            print(f"警告: 计算点到多边形距离失败 {e}")
            return float('inf')
    
    
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
        返回左右手腕的坐标
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
        公式: 中心点位置 = 0.5*left_wrist + 0.5*right_wrist
        """
        left_wrist = hand_points['left_wrist']
        right_wrist = hand_points['right_wrist']
        
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
    
    
    
    
    def detect_objects(self, frame, frame_number=0, timestamp=0.0):
        """
        使用object.json中的多边形数据作为设备位置
        
        Args:
            frame: 输入图像帧
            frame_number: 当前帧号
            timestamp: 当前时间戳(秒)（用于缓存管理）
            
        Returns:
            list: 检测到的设备列表
        """
        detected_objects = []
        
        # 获取图像尺寸
        img_height, img_width = frame.shape[:2]
        
        # 使用固定坐标置信度
        fixed_confidence = self.config['thresholds']['fixed_coordinate_confidence']
        
        print(f"🔍 帧 {frame_number}: 使用多边形数据加载设备")
        
        # 处理每个设备的多边形数据
        device_id = 0  # 分配设备ID
        for device_name, polygon_data in self.device_polygons.items():
            # 将百分比坐标转换为像素坐标
            pixel_points = self.convert_polygon_to_pixels(
                polygon_data['points_percent'], img_width, img_height
            )
            
            # 计算多边形的中心点（质心）
            center_x = sum(p[0] for p in pixel_points) / len(pixel_points)
            center_y = sum(p[1] for p in pixel_points) / len(pixel_points)
            
            # 计算边界框
            min_x = min(p[0] for p in pixel_points)
            max_x = max(p[0] for p in pixel_points)
            min_y = min(p[1] for p in pixel_points)
            max_y = max(p[1] for p in pixel_points)
            
            detected_objects.append({
                'class_id': device_id,
                'class_name': device_name,
                'center': {'x': float(center_x), 'y': float(center_y)},
                'bbox': [min_x, min_y, max_x, max_y],
                'confidence': fixed_confidence,
                'polygon_points': pixel_points,  # 添加多边形点
                'source': 'json_polygons'  # 标记来源为JSON多边形
            })
            
            device_id += 1
        
        print(f"✓ 帧 {frame_number}: 多边形模式加载了 {len(detected_objects)} 个设备")
        
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
                    hand_center = self.calculate_center_point(hand_points)
                    
                    if hand_center['x'] is not None:  # 只有当能计算出手部中心时才添加
                        detected_persons.append({
                            'person_id': i,
                            'hand_keypoints': hand_points,
                            'hand_center': hand_center
                        })
        
        return detected_persons
    
    def analyze_operations(self, persons, objects, frame_number, timestamp):
        """分析人员操作行为 - 使用多边形距离判断和时间状态管理"""
        operations = []
        operation_threshold = 60  # 固定使用60像素作为操作判断阈值
        
        for person in persons:
            if person['hand_center']['x'] is None:
                continue
            
            person_id = person['person_id']
            
            # 存储当前人员对所有设备的候选操作
            candidate_operations = []
            
            # 检查每个设备的多边形
            for obj in objects:
                device_id = obj['class_id']
                device_name = obj['class_name']
                
                # 检查手部到多边形边界的距离
                if 'polygon_points' in obj and obj['polygon_points']:
                    distance_to_polygon = self.calculate_point_to_polygon_distance(
                        person['hand_center'], 
                        obj['polygon_points']
                    )
                    
                    # 只有当距离小于60像素时才进行后续处理
                    if distance_to_polygon > operation_threshold:
                        # 距离大于60像素，检查是否需要重置状态
                        if device_id in self.operation_states[person_id]:
                            state_info = self.operation_states[person_id][device_id]
                            if state_info['status'] in ['near', 'operating']:
                                if state_info['far_start_time'] is None:
                                    # 刚开始远离
                                    state_info['far_start_time'] = timestamp
                                elif timestamp - state_info['far_start_time'] >= self.operation_end_time:
                                    # 远离时间超过2秒，结束操作
                                    state_info['status'] = 'far'
                                    state_info['near_start_time'] = None
                                    state_info['operating_start_time'] = None
                                    state_info['far_start_time'] = None
                        continue  # 跳过这个设备，不进行后续计算和记录
                    
                    # 距离小于等于60像素，初始化状态信息（如果不存在）
                    if device_id not in self.operation_states[person_id]:
                        self.operation_states[person_id][device_id] = {
                            'status': 'far',  # 状态：far/near/operating
                            'near_start_time': None,  # 开始接近的时间
                            'operating_start_time': None,  # 开始操作的时间
                            'far_start_time': None,  # 开始远离的时间
                            'last_distance': float('inf')
                        }
                    
                    state_info = self.operation_states[person_id][device_id]
                    
                    # 基于距离更新状态（只在距离<=60时执行）
                    if distance_to_polygon <= operation_threshold:
                        # 在操作范围内
                        if state_info['status'] == 'far':
                            # 从远离状态转为接近状态
                            state_info['status'] = 'near'
                            state_info['near_start_time'] = timestamp
                            state_info['far_start_time'] = None
                        elif state_info['status'] == 'near':
                            # 检查是否接近时间超过3秒
                            if timestamp - state_info['near_start_time'] >= self.operation_start_time:
                                state_info['status'] = 'operating'
                                state_info['operating_start_time'] = timestamp
                        # 如果已经在操作状态，保持状态不变
                    
                    state_info['last_distance'] = distance_to_polygon
                    
                    # 根据当前状态决定是否添加到候选操作列表
                    if state_info['status'] in ['near', 'operating']:
                        candidate_operation = {
                            'frame_number': frame_number,
                            'timestamp': timestamp,
                            'person_id': person['person_id'],
                            'device_class_id': obj['class_id'],
                            'device_name': obj['class_name'],
                            'distance': distance_to_polygon,
                            'hand_center': person['hand_center'],
                            'device_center': obj['center'],
                            'device_confidence': obj['confidence'],
                            'operation_status': state_info['status'],  # 添加操作状态
                            'near_duration': timestamp - state_info['near_start_time'] if state_info['near_start_time'] else 0,
                            'operating_duration': timestamp - state_info['operating_start_time'] if state_info['operating_start_time'] else 0,
                            'operation_type': 'time_based_polygon_detection'
                        }
                        candidate_operations.append(candidate_operation)
            
            # 每一帧每个人只能对一个设备进行交互，选择距离最近的设备
            if candidate_operations:
                # 优先选择正在操作状态的设备，如果没有则选择距离最近的接近状态设备
                operating_ops = [op for op in candidate_operations if op['operation_status'] == 'operating']
                if operating_ops:
                    best_operation = min(operating_ops, key=lambda op: op['distance'])
                else:
                    best_operation = min(candidate_operations, key=lambda op: op['distance'])
                
                operations.append(best_operation)
                
                # 只有正在操作状态的才记录到操作历史
                if best_operation['operation_status'] == 'operating':
                    self.operation_records[best_operation['device_class_id']].append(best_operation)
                
                # 打印状态信息
                status_text = {
                    'near': f"在{best_operation['device_name']}旁 (接近{best_operation['near_duration']:.1f}s)",
                    'operating': f"正在操作{best_operation['device_name']} (操作{best_operation['operating_duration']:.1f}s)"
                }
                print(f"帧 {frame_number}: 人员 {person['person_id']} {status_text.get(best_operation['operation_status'], '未知状态')} (距离: {best_operation['distance']:.1f}px)")
        
        return operations
    
    def draw_annotations(self, frame, objects, persons, operations):
        """在帧上绘制检测结果和操作分析"""
        annotated_frame = frame.copy()
        
        # 绘制检测到的设备多边形
        for obj in objects:
            center_x, center_y = int(obj['center']['x']), int(obj['center']['y'])
            
            # 绘制多边形（如果存在）
            if 'polygon_points' in obj and obj['polygon_points']:
                polygon_points = np.array(obj['polygon_points'], dtype=np.int32)
                
                # 绘制多边形边框
                polygon_color = (0, 255, 0)  # 绿色边框
                cv2.polylines(annotated_frame, [polygon_points], True, polygon_color, 2)
                
                # 绘制半透明多边形填充
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [polygon_points], (0, 255, 0))  # 绿色填充
                annotated_frame = cv2.addWeighted(annotated_frame, 0.8, overlay, 0.2, 0)
                
                # 在多边形上标注设备名称
                annotated_frame = self.draw_chinese_text(
                    annotated_frame, obj['class_name'], 
                    (center_x - 30, center_y - 10), 
                    self.chinese_font_small, 
                    (255, 255, 255)
                )
            
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
            
            # 根据操作状态选择不同的颜色和样式
            operation_status = op.get('operation_status', 'near')
            
            if operation_status == 'operating':
                # 正在操作状态：红色粗线
                line_color = (0, 0, 255)  # 红色
                line_thickness = 3
                text_color = (0, 0, 255)  # 红色文字
                operation_text = f"正在操作: {op['device_name']}"
                duration_text = f"操作时长: {op.get('operating_duration', 0):.1f}秒"
            else:
                # 接近状态：橙色细线
                line_color = (0, 165, 255)  # 橙色
                line_thickness = 2
                text_color = (0, 165, 255)  # 橙色文字
                operation_text = f"在设备旁: {op['device_name']}"
                duration_text = f"接近时长: {op.get('near_duration', 0):.1f}秒"
            
            # 绘制连线
            cv2.line(annotated_frame, (hand_x, hand_y), (device_x, device_y), line_color, line_thickness)
            
            # 标注操作信息（使用中文字体）
            mid_x = (hand_x + device_x) // 2
            mid_y = (hand_y + device_y) // 2
            
            # 显示到多边形边界的距离
            if op['distance'] == 0.0:
                distance_text = f"边界距离: 内部/边界上"
            else:
                distance_text = f"边界距离: {op['distance']:.1f}px"
            
            # 使用中文字体绘制操作信息
            annotated_frame = self.draw_chinese_text(
                annotated_frame, operation_text, (mid_x, mid_y-35), self.chinese_font_small, text_color
            )
            annotated_frame = self.draw_chinese_text(
                annotated_frame, duration_text, (mid_x, mid_y-15), self.chinese_font_small, text_color
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
        
        
        print(f"视频信息: {width}x{height}, {fps} FPS, 共 {total_frames} 帧")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_dir, 'operation_analysis.mp4')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 数据记录
        all_frame_data = []
        frame_count = 0
        processed_count = 0  # 实际处理的帧数
        
        # 缓存最后一次的检测结果，用于跳帧时继续显示标注
        last_detected_objects = []
        last_detected_persons = []
        last_operations = []
        
        # 计算帧间隔：每秒10帧 = fps/10 帧间隔
        frame_interval = max(1, fps // 10)  # 确保至少为1
        print(f"检测设置: 每秒检测3帧 (每 {frame_interval} 帧检测一次)")
        print(f"标注显示: 所有帧都显示标注 (使用最新检测结果)")
        
        print("开始处理视频...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            if frame_count % 300 == 0:  # 每300帧显示进度
                print(f"处理进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) - 已检测 {processed_count} 帧")
            
            # 只在指定间隔的帧上进行检测和分析
            if (frame_count - 1) % frame_interval == 0:
                processed_count += 1
                
                # 检测物体和人体姿态
                detected_objects = self.detect_objects(frame, frame_count, timestamp)
                detected_persons = self.detect_poses(frame)
                
                # 分析操作行为
                operations = self.analyze_operations(detected_persons, detected_objects, frame_count, timestamp)
                
                # 缓存当前检测结果
                last_detected_objects = detected_objects
                last_detected_persons = detected_persons
                last_operations = operations
                
                # 绘制标注
                annotated_frame = self.draw_annotations(frame, detected_objects, detected_persons, operations)
            else:
                # 跳过的帧使用缓存的检测结果来绘制标注，保持标注连续显示
                if last_detected_objects and last_detected_persons:
                    annotated_frame = self.draw_annotations(frame, last_detected_objects, last_detected_persons, last_operations)
                else:
                    # 如果还没有缓存结果，使用原帧
                    annotated_frame = frame
            
            # 写入视频
            out_video.write(annotated_frame)
        
        cap.release()
        out_video.release()
        
        print(f"视频处理完成！共 {frame_count} 帧，实际检测 {processed_count} 帧 (每秒3帧)")
        print(f"检测效率提升: {frame_count/processed_count:.1f}x 速度")
        print(f"标注显示: 所有帧都有连续标注显示")
        print(f"标注视频保存为: {output_video_path}")
        
        return all_frame_data
    
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
        detector.process_video(video_path, output_dir)
        
        print("✅ 分析完成！")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
