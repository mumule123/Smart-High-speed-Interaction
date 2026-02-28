# 设备操作检测系统 (Equipment Operation Detection System)

## 项目简介

本项目是一个基于计算机视觉的电力设备操作检测系统。通过使用YOLOv11模型进行人体姿态估计，结合预定义的设备区域信息，自动分析视频中的人员操作行为。系统能够识别人员是否在特定设备区域进行操作，并生成相应的分析报告。

## 功能特性

- **人体姿态识别**：使用YOLOv11-pose模型精确检测人体关键点（主要关注手腕位置）。
- **设备区域定义**：支持通过JSON文件定义复杂的设备多边形区域，适应各种不规则设备形状。
- **操作行为判定**：基于手部关键点与设备区域的距离/包含关系，智能判断操作状态（开始、进行中、结束）。
- **可配置化**：通过 `config.yaml` 灵活配置模型路径、判定阈值、权重参数等。
- **结果分析**：自动生成操作统计报告（JSON）和数据表格（CSV）。

## 目录结构说明

```
/t-3058/pythonProject02/yoloV11/video_new_test/
├── config.yaml              # 系统主配置文件 (包含模型路径、阈值、参数设置)
├── src/                     # ✅ 源代码目录
│   ├── config_manager.py    # 配置管理器
│   ├── main.py              # ✅ 主检测程序 (原 detectV5_json.py)
│   └── visualizer.py        # 视频可视化展示脚本 (原 video_object_display.py)
├── archive/                 # 归档目录 (旧版本脚本及备份)
│   ├── detectV1.py ...      
│   └── run.txt
├── datasets/                # 数据集与元数据
│   ├── object.json          # 设备区域标注数据 (多边形坐标)
│   ├── classes.txt          # 检测类别列表
│   └── [视频文件夹]
├── model/                   # 模型权重文件
├── font/                    # 字体文件
├── analysis_resultsv3/      # 分析结果输出目录
├── results/                 # 视频处理结果输出目录
└── requirements.txt         # Python依赖列表
```

## 核心配置 (config.yaml)

系统行为可以通过 `config.yaml` 进行调整，主要参数包括：

*   **Models**: 指定检测 (`best.pt`) 和姿态 (`yolo11l-pose.pt`) 模型路径。
*   **Thresholds**:
    *   `operation_distance_threshold`: 判定操作的距离阈值 (默认 120.0 像素)。
    *   `detection_confidence_threshold`: 目标检测置信度。
    *   `keypoint_confidence_threshold`: 关键点置信度。
*   **Hand Center Weights**: 计算手部中心点时左右手腕的权重分配。

## 如何运行

主要运行脚本为 `src/main.py`。

### 环境依赖

首先安装项目依赖：

```bash
pip install -r requirements.txt
```

### 运行示例

在项目根目录下运行：

```bash
python src/main.py
```
(注意：脚本内部可能需要根据实际情况调整输入视频路径或通过命令行参数传入)

## 输出结果

程序运行主要产生两类结果：
1.  **可视化的视频/图像**：在 `results/` 目录下，视频中会标注出人体骨架、设备区域以及实时的操作状态。
2.  **数据分析报告**：在 `analysis_resultsv3/` 目录下：
    *   `analysis_report.json`: 详细记录了每个操作事件的时间段、操作人、设备等。
    *   `device_operation_stats.csv`: 统计表格，方便进行数据汇总。

## 注意事项
- 确保 `datasets/object.json`文件存在且格式正确，它定义了视频中设备的位置。
- 首次运行时，如果模型文件不存在，Ultralytics可能会尝试自动下载标准模型。
