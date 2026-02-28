#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器模块
负责处理所有配置文件相关的功能，包括：
- YAML配置文件加载
- 颜色格式转换 (RGB/HEX -> BGR)
- 参数验证和范围检查
- 默认值处理
"""

import yaml
import os
from typing import Dict, Any, Union, List, Tuple


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self):
        """初始化配置管理器"""
        # 预定义颜色名称映射 (RGB格式)
        self.color_names = {
            'red': [255, 0, 0],
            'green': [0, 255, 0], 
            'blue': [0, 0, 255],
            'yellow': [255, 255, 0],
            'cyan': [0, 255, 255],
            'magenta': [255, 0, 255],
            'white': [255, 255, 255],
            'black': [0, 0, 0],
            'orange': [255, 165, 0],
            'purple': [128, 0, 128],
            'pink': [255, 192, 203],
            'lime': [0, 255, 0],
            'navy': [0, 0, 128],
            'maroon': [128, 0, 0],
            'olive': [128, 128, 0],
            'teal': [0, 128, 128],
            'gray': [128, 128, 128],
            'grey': [128, 128, 128],
            'silver': [192, 192, 192],
            'gold': [255, 215, 0]
        }
    
    def parse_color(self, color_value: Union[str, List[int], Tuple[int]]) -> Tuple[int, int, int]:
        """
        解析颜色值，输出BGR格式供给OpenCV使用
        
        支持的格式:
        1. RGB列表: [255, 0, 0] -> 红色
        2. 16进制字符串: "#FF0000" 或 "FF0000" -> 红色  
        3. 颜色名称: "red", "green", "blue" 等
        
        Args:
            color_value: 颜色值，RGB列表、16进制字符串或颜色名称
            
        Returns:
            tuple: BGR格式的颜色值 (B, G, R)
        """
        try:
            # 情况1: 颜色名称
            if isinstance(color_value, str) and color_value.lower() in self.color_names:
                r, g, b = self.color_names[color_value.lower()]
                return (b, g, r)  # 转换为BGR
            
            # 情况2: 16进制字符串
            if isinstance(color_value, str):
                hex_color = color_value.lstrip('#').upper()
                if len(hex_color) == 6 and all(c in '0123456789ABCDEF' for c in hex_color):
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    return (b, g, r)  # 转换为BGR
                elif len(hex_color) == 3 and all(c in '0123456789ABCDEF' for c in hex_color):
                    # 支持3位16进制 如 #F0A -> #FF00AA
                    r = int(hex_color[0] * 2, 16)
                    g = int(hex_color[1] * 2, 16)
                    b = int(hex_color[2] * 2, 16)
                    return (b, g, r)  # 转换为BGR
            
            # 情况3: RGB列表或元组
            if isinstance(color_value, (list, tuple)) and len(color_value) == 3:
                r, g, b = color_value
                # 验证颜色值范围
                if all(0 <= c <= 255 for c in [r, g, b]):
                    return (int(b), int(g), int(r))  # 转换为BGR
                else:
                    raise ValueError(f"RGB颜色值必须在0-255范围内: {color_value}")
            
            # 如果都不匹配，报错
            raise ValueError(f"不支持的颜色格式: {color_value}")
            
        except Exception as e:
            print(f"⚠️  颜色解析失败 {color_value}: {e}")
            print("   使用默认红色")
            return (0, 0, 255)  # 默认红色 (BGR)
    
    def validate_threshold(self, value: float, min_val: float, max_val: float, name: str) -> float:
        """
        验证阈值参数范围
        
        Args:
            value: 要验证的值
            min_val: 最小值
            max_val: 最大值  
            name: 参数名称
            
        Returns:
            float: 验证后的值
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} 必须是数字类型，得到: {type(value)}")
        
        if not (min_val <= value <= max_val):
            print(f"⚠️  {name} 超出范围 [{min_val}, {max_val}]，当前值: {value}")
            # 自动修正到范围内
            value = max(min_val, min(max_val, value))
            print(f"   已自动修正为: {value}")
        
        return float(value)
    
    def validate_integer(self, value: Union[int, float], min_val: int, max_val: int, name: str) -> int:
        """验证整数参数范围"""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} 必须是数字类型，得到: {type(value)}")
        
        int_value = int(value)
        if not (min_val <= int_value <= max_val):
            print(f"⚠️  {name} 超出范围 [{min_val}, {max_val}]，当前值: {int_value}")
            int_value = max(min_val, min(max_val, int_value))
            print(f"   已自动修正为: {int_value}")
        
        return int_value
    
    def process_colors_in_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理配置中的所有颜色值，转换为BGR格式
        
        Args:
            config: 原始配置字典
            
        Returns:
            Dict: 处理后的配置字典
        """
        # 处理可视化颜色
        if 'visualization' in config:
            viz_colors = ['device_center_color', 'hand_center_color', 'operation_line_color', 'operation_text_color']
            for color_key in viz_colors:
                if color_key in config['visualization']:
                    original = config['visualization'][color_key]
                    config['visualization'][color_key] = list(self.parse_color(original))
        
        # 处理手部关键点颜色
        if 'hand_colors' in config:
            for hand_part, color_value in config['hand_colors'].items():
                original = color_value
                config['hand_colors'][hand_part] = list(self.parse_color(original))
        
        return config
    
    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        验证手部权重配置，允许权重大于1，会自动归一化
        
        Args:
            weights: 权重配置字典
            
        Returns:
            Dict: 验证后的权重配置（已归一化）
        """
        required_keys = ['left_wrist_weight', 'right_wrist_weight']
        
        # 检查必需的键并验证非负数
        for key in required_keys:
            if key not in weights:
                raise ValueError(f"缺少必需的权重配置: {key}")
            
            # 验证权重为非负数
            if not isinstance(weights[key], (int, float)) or weights[key] < 0:
                print(f"⚠️  {key} 必须是非负数，当前值: {weights[key]}")
                weights[key] = 0.25  # 设置默认值
                print(f"   已修正为: 0.25")
            else:
                weights[key] = float(weights[key])
        
        # 计算权重总和
        total_weight = sum(weights[key] for key in required_keys)
        
        if total_weight == 0:
            print("⚠️  所有权重都为0，使用默认权重配置")
            weights['left_wrist_weight'] = 0.35
            weights['right_wrist_weight'] = 0.35
            weights['left_elbow_weight'] = 0.15
            weights['right_elbow_weight'] = 0.15
            total_weight = 1.0
        
        # 如果总和不为1，自动归一化
        if abs(total_weight - 1.0) > 0.001:  # 允许小的浮点误差
            print(f"🔄 手部权重总和为 {total_weight:.3f}，自动归一化到1.0")
            factor = 1.0 / total_weight
            for key in required_keys:
                old_value = weights[key]
                weights[key] = weights[key] * factor
                print(f"   {key}: {old_value:.3f} -> {weights[key]:.3f}")
        
        return weights
    
    def validate_config_structure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证和修正配置文件结构
        
        Args:
            config: 原始配置字典
            
        Returns:
            Dict: 验证后的配置字典
        """
        # 验证阈值参数
        if 'thresholds' in config:
            thresholds = config['thresholds']
            
            # 操作距离阈值 - 只要求大于0即可
            if 'operation_distance_threshold' in thresholds:
                value = thresholds['operation_distance_threshold']
                if not isinstance(value, (int, float)) or value <= 0:
                    print(f"⚠️  操作距离阈值必须大于0，当前值: {value}")
                    thresholds['operation_distance_threshold'] = 120.0  # 使用默认值
                    print(f"   已自动修正为: 120.0")
                else:
                    thresholds['operation_distance_threshold'] = float(value)
            
            # 检测置信度阈值
            if 'detection_confidence_threshold' in thresholds:
                thresholds['detection_confidence_threshold'] = self.validate_threshold(
                    thresholds['detection_confidence_threshold'], 0.01, 0.99, '检测置信度阈值'
                )
            
            # 关键点置信度阈值
            if 'keypoint_confidence_threshold' in thresholds:
                thresholds['keypoint_confidence_threshold'] = self.validate_threshold(
                    thresholds['keypoint_confidence_threshold'], 0.01, 0.99, '关键点置信度阈值'
                )
        
        # 验证手部权重
        if 'hand_center_weights' in config:
            config['hand_center_weights'] = self.validate_weights(config['hand_center_weights'])
        
        # 验证设备缓存配置
        if 'device_cache' in config:
            cache_config = config['device_cache']
            
            # 验证缓存最大存活时间
            if 'max_age_seconds' in cache_config:
                cache_config['max_age_seconds'] = self.validate_threshold(
                    cache_config['max_age_seconds'], 1.0, 300.0, '缓存最大存活时间(秒)'
                )
            
            # 兼容旧的帧数配置，但不推荐
            if 'max_age_frames' in cache_config:
                print("⚠️  'max_age_frames' 配置已废弃，请使用 'max_age_seconds'")
                if 'max_age_seconds' not in cache_config:
                    # 假设30fps转换为秒数
                    cache_config['max_age_seconds'] = cache_config['max_age_frames'] / 30.0
                    print(f"   已自动转换为: {cache_config['max_age_seconds']}秒")
                del cache_config['max_age_frames']
            
            # 验证最低缓存置信度
            if 'min_confidence_to_cache' in cache_config:
                cache_config['min_confidence_to_cache'] = self.validate_threshold(
                    cache_config['min_confidence_to_cache'], 0.1, 0.95, '最低缓存置信度阈值'
                )
            
            # 验证位置平滑系数
            if 'cache_update_smoothing' in cache_config:
                cache_config['cache_update_smoothing'] = self.validate_threshold(
                    cache_config['cache_update_smoothing'], 0.1, 0.99, '位置更新平滑系数'
                )
        
        # 验证可视化参数
        if 'visualization' in config:
            viz = config['visualization']
            
            # 验证半径参数
            if 'device_center_radius' in viz:
                viz['device_center_radius'] = self.validate_integer(
                    viz['device_center_radius'], 1, 50, '设备中心点半径'
                )
            
            if 'hand_center_radius' in viz:
                viz['hand_center_radius'] = self.validate_integer(
                    viz['hand_center_radius'], 1, 50, '手部中心点半径'
                )
            
            if 'keypoint_radius' in viz:
                viz['keypoint_radius'] = self.validate_integer(
                    viz['keypoint_radius'], 1, 30, '关键点半径'
                )
            
            if 'operation_line_thickness' in viz:
                viz['operation_line_thickness'] = self.validate_integer(
                    viz['operation_line_thickness'], 1, 20, '操作连线粗细'
                )
        
        # 验证视频处理参数
        if 'video_processing' in config:
            video = config['video_processing']
            
            if 'fps_assumption' in video:
                video['fps_assumption'] = self.validate_integer(
                    video['fps_assumption'], 1, 240, '回退帧率'
                )
            
            if 'progress_report_interval' in video:
                video['progress_report_interval'] = self.validate_integer(
                    video['progress_report_interval'], 1, 1000, '进度报告间隔'
                )
        
        return config
    
    def load_config(self, config_file_path: str) -> Dict[str, Any]:
        """
        加载并处理YAML配置文件
        
        Args:
            config_file_path: 配置文件路径
            
        Returns:
            Dict: 处理后的配置字典
        """
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"配置文件不存在: {config_file_path}")
        
        try:
            # 加载YAML文件
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ YAML配置文件加载成功: {config_file_path}")
            
            # 验证配置结构
            config = self.validate_config_structure(config)
            print("✅ 配置参数验证完成")
            
            # 处理颜色转换
            config = self.process_colors_in_config(config)
            print("✅ 颜色转换完成 (RGB/HEX -> BGR)")
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"YAML格式错误: {e}")
        except Exception as e:
            raise ValueError(f"配置文件处理失败: {e}")
    
    def get_color_examples(self) -> str:
        """
        获取颜色格式示例
        
        Returns:
            str: 颜色格式示例说明
        """
        examples = """
🎨 支持的颜色格式示例:

1. RGB列表格式:
   red: [255, 0, 0]
   green: [0, 255, 0]
   blue: [0, 0, 255]

2. 16进制格式:
   red: "#FF0000"
   green: "#00FF00"  
   blue: "#0000FF"
   简写: "#F00", "#0F0", "#00F"

3. 颜色名称:
   red, green, blue, yellow, cyan, magenta
   white, black, orange, purple, pink, lime
   navy, maroon, olive, teal, gray, silver, gold

注意: 所有颜色会自动转换为OpenCV需要的BGR格式
        """
        return examples


# 便利函数
def load_config(config_file_path: str) -> Dict[str, Any]:
    """
    便利函数：加载配置文件
    
    Args:
        config_file_path: 配置文件路径
        
    Returns:
        Dict: 处理后的配置字典
    """
    manager = ConfigManager()
    return manager.load_config(config_file_path)


def show_color_examples():
    """显示颜色格式示例"""
    manager = ConfigManager()
    print(manager.get_color_examples())


if __name__ == "__main__":
    # 测试配置管理器
    show_color_examples()