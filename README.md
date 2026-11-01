# YOLO-Export-Inferencce
# YOLO模型导出、推理与性能对比

## 1.模型版本
YOLO26n 装甲板关键点检测模型，支持识别B1/B2/B3装甲板，每个目标输出4个二维关键点，原始权重为 PyTorch .pt 格式，输入尺寸 640×640。

## 2.导出的模型格式及方法
本次导出 **ONNX** 格式：
- 格式特点：通用模型交换格式，跨平台、跨框架兼容，是工业部署的标准中间格式
- 导出工具：Ultralytics 原生 export 接口，开启计算图简化，opset=12
- 导出命令：`python export.py`
- 导出产物：`models/best.onnx`

## 3.运行环境与依赖库
- 操作系统：Windows 11
- 运行硬件：CPU 推理
- Python版本：3.12
- 核心依赖：ultralytics、onnx、onnxruntime、opencv-python、numpy

## 4.模型推理运行命令
```bash
# 安装依赖
pip install -r requirements.txt

# 导出ONNX模型
python export.py

# 双模型推理+性能对比（自动跑PT和ONNX）
python inference.py
```

## 5.推理结果截图
<img width="375" height="344" alt="屏幕截图 2026-11-01 215015" src="https://github.com/user-attachments/assets/c6617356-c2d0-4f62-ba00-ba3068457d86" />


## 6.pt与onnx对比
| 对比项目 | 原始 .pt 模型 | 导出后的模型 |
|---|---|---|
| 模型格式 | .pt | ONNX |
| 模型文件大小 | 5.62MB | 9.72MB |
| 平均推理耗时 | 36.55ms | 48.48ms |
| 检测目标数量 | 1.29 | 1.30 |

## 7.实验过程中遇到的问题及解决方法
- 1. 缺少 onnx 依赖 → 安装 onnx 和 onnxruntime 解决
- 2. 首次推理耗时偏高 → 增加 10 帧预热，排除初始化开销后统计
- 3. 结果存在微小数值差异 → 属于浮点计算正常误差，不影响实际检测
- 4. GitHub 推送失败 → 切换手机热点 / 使用镜像代理解决网络问题

## 8.格式理解与实验结论

- .pt 是训练原生格式，包含完整权重与结构，适合调试调优，但强依赖 PyTorch 环境，部署局限性大
- ONNX 是工业界通用部署格式，跨语言、跨硬件、跨平台，推理引擎有专门优化，精度损失极小
