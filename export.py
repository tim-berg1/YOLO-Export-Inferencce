from ultralytics import YOLO
import os

def main():
    pt_path = "./models/best.pt"
    if not os.path.exists(pt_path):
        raise FileNotFoundError("请将best.pt放入models文件夹")
    
    model = YOLO(pt_path)
    
    # 导出ONNX格式
    print("正在导出ONNX模型...")
    onnx_path = model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        opset=12,
        dynamic=False
    )
    
    # 打印文件大小（对比表要用）
    pt_size = os.path.getsize(pt_path) / 1024 / 1024
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    print("\n===== 模型文件大小 =====")
    print(f"原始PT模型: {pt_size:.2f} MB")
    print(f"ONNX模型:   {onnx_size:.2f} MB")

if __name__ == "__main__":
    main()
