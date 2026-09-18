# https://docs.ultralytics.com/
#
# 导出 YOLOv8 ONNX 模型，供 TensorRT 构建 engine 使用
#
# 用法:
#   python export_yolo.py                                # 默认导出 yolov8n-seg (多任务: 检测+分割)
#   python export_yolo.py -m yolov8n.pt                  # 导出检测模型
#   python export_yolo.py -m yolov8s-seg.pt --imgsz 640  # 指定模型与输入尺寸
#
# 输出布局（以检测模型为例）:
#   input:  1 * 3 * 640 * 640
#   output: 1 * 84 * 8400 (80 classes + 4 coordinates[cx, cy, w, h])
#
# 说明:
#   - 推荐使用 n/s 轻量规格做流程验证，INT8 引擎构建耗时可控；
#   - 需要环境: pip install ultralytics（自动携带 torch/onnx）

import argparse
import os
import shutil

import onnx
from onnx import helper

from ultralytics import YOLO


def transpose_box_output(onnx_path):
    """将检测头输出从 (1, 4+cls+mc, 8400) 转置为 (1, 8400, 4+cls+mc)

    ultralytics 默认导出的 output0 通道在前，而本项目 C++ 后处理解码
    假设每个 box 的 [cx, cy, w, h, cls..., mc...] 连续存放，
    因此在图末尾追加 Transpose(0,2,1) 节点。
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    for i, out in enumerate(graph.output):
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        if len(dims) != 3:
            continue  # 只处理 3 维的检测头输出，跳过分割 proto 等其它输出

        old_name = out.name
        new_name = old_name + "_boxmajor"
        graph.node.append(helper.make_node(
            "Transpose", [old_name], [new_name], perm=[0, 2, 1],
            name="transpose_box_output"))
        new_vi = helper.make_tensor_value_info(
            new_name, out.type.tensor_type.elem_type, [dims[0], dims[2], dims[1]])
        graph.output[i].CopyFrom(new_vi)
        print(f"Transposed output '{old_name}' {dims} -> '{new_name}' {[dims[0], dims[2], dims[1]]}")

    onnx.save(model, onnx_path)


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX")
    parser.add_argument("-m", "--model", default="yolov8n-seg.pt",
                        help="ultralytics 模型名或本地权重路径 (默认: yolov8n-seg.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸 (默认: 640)")
    parser.add_argument("-o", "--output-dir", default=os.path.join(os.path.dirname(__file__), "onnx"),
                        help="ONNX 输出目录 (默认: data/models/onnx)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(args.model)

    # Export the model to ONNX format, ultralytics 会在权重旁生成同名 .onnx
    onnx_path = model.export(format="onnx", imgsz=args.imgsz)

    # 检测头输出转置为 box-major，适配 C++ 后处理解码
    transpose_box_output(onnx_path)

    # 统一收拢到输出目录
    target = os.path.join(args.output_dir, os.path.basename(onnx_path))
    if os.path.abspath(onnx_path) != os.path.abspath(target):
        shutil.move(onnx_path, target)
    print(f"ONNX saved to: {target}")


if __name__ == "__main__":
    main()
