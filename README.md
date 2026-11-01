# YOLO-Export-Inferencce
# YOLO模型导出、推理与性能对比

## 1. 模型版本
YOLO26n 装甲板关键点检测模型，支持识别B1/B2/B3装甲板，每个目标输出4个二维关键点，原始权重为 PyTorch .pt 格式，输入尺寸 640×640。

## 2. 导出的模型格式及方法
本次导出 **ONNX** 格式：
- 格式特点：通用模型交换格式，跨平台、跨框架兼容，是工业部署的标准中间格式
- 导出工具：Ultralytics 原生 export 接口，开启计算图简化，opset=12
- 导出命令：`python export.py`
- 导出产物：`models/best.onnx`

## 3. 运行环境与依赖库
- 操作系统：Windows 11
- 运行硬件：CPU 推理
- Python版本：3.12
- 核心依赖：ultralytics、onnx、onnxruntime、opencv-python、numpy

## 4. 模型推理运行命令
```bash
# 安装依赖
pip install -r requirements.txt

# 导出ONNX模型
python export.py

# 双模型推理+性能对比（自动跑PT和ONNX）
python inference.py
