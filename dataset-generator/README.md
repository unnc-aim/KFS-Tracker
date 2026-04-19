# wulin-recognition

3D 空间变换 OCR 数据集生成与可视化系统（FastAPI + OpenCV + SQLite）。

## 功能

- 原始样本上传（图片 + 文本 + bbox）
- 批量上传（支持文件名自动作为文本标签）
- 批量拉框并保存 bbox
- 3D 透视增强（Pitch/Yaw/Roll + 平移 + 缩放 + 焦距）
- 标注重映射并输出 YOLO 四边形标签（class + 8 点归一化坐标）
- 可视化页面展示增强结果与检测框
- 一键导出 YOLO 数据集 zip（images + labels）
- 一键导出带框图片 zip（boxed_images）
- 自动化测试（算法 + API 端到端）

## 本地运行

1. 安装依赖
2. 启动服务：`uvicorn app.main:app --reload`
3. 打开：`http://127.0.0.1:8000/`

## API 摘要

- `POST /api/v1/samples/upload`
- `POST /api/v1/samples/batch-upload`
- `GET /api/v1/samples`
- `PATCH /api/v1/samples/batch-bbox`
- `POST /api/v1/augment/jobs`
- `GET /api/v1/augment/jobs/{job_id}`
- `GET /api/v1/generated`
- `GET /api/v1/generated/{gen_id}/image`
- `GET /api/v1/generated/{gen_id}/label`
- `PATCH /api/v1/generated/{gen_id}/review`
- `POST /api/v1/export/yolo`
- `POST /api/v1/export/boxed-images`
- `GET /api/v1/export/yolo/{export_job_id}/download`

## 标签格式

- 新格式（默认）：`class x1 y1 x2 y2 x3 y3 x4 y4`（归一化到 [0,1]）
- 顶点顺序：顺时针，起点尽量为左上邻近点

## 输出命名与目录

- 生成图像：`data/generated/images/gen_<文本标签>_sid<样本ID>_*.jpg`
- 生成标签：`data/generated/data/gen_<文本标签>_sid<样本ID>_*.txt`
- 带框导出：`boxed_images/boxed_<序号>_<文本标签>.jpg`

说明：文件名会对文本标签做安全清洗（保留中英文、数字、`_`、`-`）。

## generated 归类脚本

新增脚本：`scripts/classify_generated.py`

用途：将 `data/generated/images` 与 `data/generated/data` 中同名文件按标签归类到同一目录，结构如下：

- `<标签>/<学校或战队名>/frame_000001.jpg`
- `<标签>/<学校或战队名>/frame_000001.txt`

例如：`R_T_8/UNNC/frame_000001.jpg` 与 `R_T_8/UNNC/frame_000001.txt`

默认行为：

- 输入目录：`data/generated`
- 输出目录：`data/generated/classified`
- 模式：`copy`（复制，不改动原文件）

可选参数：

- `--team-name UNNC`：直接指定学校缩写/战队名；不传时会交互提示输入
- `--mode move`：移动文件而非复制
- `--dry-run`：仅预览，不实际写入
