# hypn-player-detect

Minecraft PvP **Player** 目标检测：数据集准备、远程 GPU 微调、ONNX 推理与视频标注 GUI。

## 功能

- 从 LabelMe / X-AnyLabeling JSON 生成 YOLO 数据集（train / val / test）
- 上传 Seetacloud 等远程 GPU 训练 YOLO
- 导出 **ONNX** + **X-AnyLabeling** 用 `best.yaml`
- 本地 **DirectML (AMD GPU)** 视频检测 GUI

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

将标注 JSON 与对应 JPG 放在源目录后运行：

```bash
python prepare_player_dataset.py
```

输出目录：`player_dataset/`（labels + `data.yaml`）

### 3. 视频检测 GUI

将训练好的 `best.onnx` 放到 `runs/player_finetune/weights/`，然后：

```bash
run_player_gui.bat
```

### 4. 远程训练（可选）

设置 SSH 密码环境变量（不要写进代码）：

```powershell
$env:SSH_PASSWORD = "your_password"
python full_train_pipeline.py
```

一键完成：打包数据集 → 上传 → 训练 → 导出 ONNX/YAML → 下载权重。

## X-AnyLabeling

加载 `runs/player_finetune/weights/best.yaml`（不是 `.onnx`）。  
`model_path` 指向同目录下的 `best.onnx`，输入尺寸 320×320，类别 `Player`。

## 目录结构

```
├── prepare_player_dataset.py   # JSON → YOLO 数据集
├── full_train_pipeline.py      # 远程训练全流程
├── player_onnx_detector.py     # ONNX + DirectML 检测
├── player_video_gui.py         # 视频标注 GUI
├── player_dataset/             # 数据集（labels，images 需本地生成）
└── runs/player_finetune/weights/
    ├── best.yaml               # X-AnyLabeling 配置模板
    ├── best.pt                 # 本地训练产物（gitignore）
    └── best.onnx               # 本地导出产物（gitignore）
```

## 环境变量

| 变量 | 用途 |
|------|------|
| `SSH_PASSWORD` | 远程 GPU 服务器 SSH 密码 |

复制 `.env.example` 为本地参考，**不要提交 `.env`**。

## 说明

- 模型权重（`.pt` / `.onnx`）和大体积数据集图片默认在 `.gitignore` 中，需自行训练或下载后放置。
- 首次 commit 历史若含本地调试脚本，请以当前仓库版本为准；敏感信息请使用环境变量。

## License

MIT
