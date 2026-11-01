import cv2
import time
import os
from ultralytics import YOLO

def run_inference(model_path, input_video, output_path, conf=0.3):
    print(f"\n========== 测试模型：{os.path.basename(model_path)} ==========")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError("找不到输入视频")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    warmup_frames = 10   # 预热帧，不计入统计
    test_frames = 100    # 统计帧数
    frame_id = 0
    total_time = 0.0
    total_dets = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        t1 = time.time()
        res = model(frame, conf=conf, verbose=False)[0]
        t2 = time.time()
        infer_ms = (t2 - t1) * 1000
        
        # 预热阶段：只画图，不统计
        if frame_id < warmup_frames:
            frame_id += 1
            writer.write(res.plot())
            continue
        
        # 统计阶段：累计耗时和检测数
        if frame_id < warmup_frames + test_frames:
            total_time += infer_ms
            if res.boxes is not None:
                total_dets += len(res.boxes)
            frame_id += 1
        
        # 全程绘制结果保存视频
        writer.write(res.plot())
    
    cap.release()
    writer.release()
    
    valid_frames = min(test_frames, frame_id - warmup_frames)
    avg_time = total_time / valid_frames
    avg_dets = total_dets / valid_frames
    
    print(f"有效统计帧数: {valid_frames}")
    print(f"平均单帧推理耗时: {avg_time:.2f} ms")
    print(f"平均单帧检测目标数: {avg_dets:.2f}")
    print(f"结果视频已保存: {output_path}")
    return avg_time, avg_dets

if __name__ == "__main__":
    input_video = "input_video.mp4"
    conf_threshold = 0.3
    
    # 1. 原始PT模型测试
    pt_time, pt_dets = run_inference(
        "./models/best.pt",
        input_video,
        "./results/result_pt.mp4",
        conf_threshold
    )
    
    # 2. ONNX模型测试
    onnx_time, onnx_dets = run_inference(
        "./models/best.onnx",
        input_video,
        "./results/result_onnx.mp4",
        conf_threshold
    )
    
    # 最终对比汇总
    print("\n========== 最终对比汇总 ==========")
    print(f"格式        平均耗时(ms)    平均检测数")
    print(f"PT模型      {pt_time:.2f}          {pt_dets:.2f}")
    print(f"ONNX模型    {onnx_time:.2f}          {onnx_dets:.2f}")
