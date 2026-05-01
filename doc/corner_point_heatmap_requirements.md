# 需求文档：Corner Point 从坐标回归改为 Heatmap 预测

## 1. 目标

将 `corner_point_dataset` 与训练流程从“直接回归 4 个角点的 `(x, y)` 坐标”改为“预测 4 个角点 heatmap”。

- 数据来源保持不变：仍使用传统 CV 方法 `detect_colored_quad_corners(...)` 生成角点。
- 新增步骤：把传统 CV 角点转成 4 通道热力图标签。
- 模型改为类 U-Net 结构。
- 输出改为 `4` 通道，每个通道对应一个角点的 heatmap。

## 2. 范围与不变项

### 2.1 保持不变

- 数据根目录结构不变。
- 传统检测函数 `detect_colored_quad_corners(...)` 逻辑与成功判定标准不变。
- 输入图像统一尺寸策略（如现有 letterbox 填充）可继续沿用。

### 2.2 需要改造

- `corner_point_dataset.py`：标签从 `corners_xy_norm (8,)` 改为 `heatmaps (4, H, W)`（可同时保留坐标用于可视化评估）。
- 模型：改为 U-Net 风格编码器-解码器，输出 4 通道 heatmap。
- `corner_point_train.py`：loss 从坐标回归切换为 heatmap loss，并保留分类支路联合训练。

## 3. 数据改造步骤（Dataset）

## 3.1 样本生成流程

1. 读取图像并做统一尺寸处理（等比缩放 + 黑边填充）。
2. 调用 `detect_colored_quad_corners(image_bgr, color)`。
3. 若返回 `true`，拿到 4 个归一化角点坐标。
4. 将归一化坐标映射到 heatmap 网格坐标。
5. 为每个角点生成 1 张二维高斯图，共 4 张，堆叠成 `shape=(4,H,W)`。

## 3.2 Heatmap 生成规范

- 每个通道只对应一个角点（固定顺序：0/1/2/3）。
- 高斯中心为角点像素位置，标准差建议 `sigma=2~4`（按输出分辨率调参）。
- 值域建议 `[0,1]`，峰值为 `1`。
- 边界处高斯允许截断。

## 3.3 数据返回字段建议

- `target["heatmaps"]`: `torch.float32`, `shape=(4,H,W)`
- `target["class_id"]`: 分类标签（保持现有）
- 可选保留：
  - `target["corners_xy_norm"]`（用于可视化/评估）
  - `target["label_name"]`、`image_path`

## 4. 模型改造步骤（类 U-Net）

## 4.1 结构要求

- 编码器：多层卷积 + 下采样，提取多尺度语义。
- 解码器：上采样 + 跳连（skip connection）融合细节。
- 输出头：
  - `heatmap_head`: `Conv2d(..., out_channels=4, kernel_size=1)`
  - `classification_head`: 保留全局分类分支（可接编码器最深层特征）

## 4.2 输出定义

- `pred_heatmaps`: `shape=(B,4,H,W)`
- `pred_class_logits`: `shape=(B,num_classes)`

## 5. 训练改造步骤

## 5.1 Loss 设计

- 定位损失：`L_heatmap`（推荐 `MSELoss` 或 `BCEWithLogitsLoss`）
- 分类损失：`L_cls = CrossEntropyLoss`
- 总损失：`L = w_hm * L_heatmap + w_cls * L_cls`

要求：`w_hm` 与 `w_cls` 设置为平衡两任务贡献（可先设 `1:1`，再根据统计量微调）。

## 5.2 训练循环改造点

1. `extract_targets` 改为提取 `heatmaps`。
2. 前向输出改为 `(pred_heatmaps, pred_class_logits)`。
3. 评估指标新增：
   - 分类准确率 `accuracy`
   - 角点定位指标（可选）：从 heatmap 解码角点后计算平均像素误差

## 6. 推理与可视化改造步骤

## 6.1 Heatmap 解码

- 对每个通道做 `argmax` 得到角点坐标（可选 soft-argmax 提升亚像素精度）。
- 将坐标映射回原图尺寸。

## 6.2 可视化

1. 在原图画 4 个角点与连线轮廓。
2. 标注分类结果与置信度。
3. 可选输出每个角点的 heatmap 叠加图，便于调试。

## 7. 实施顺序建议

1. 先改 Dataset，完成 heatmap 标签生成并通过样本可视化检查。
2. 再改模型为 U-Net 输出 4 通道 + 分类分支。
3. 最后改训练/评估与推理脚本。
4. 补充一个单图可视化入口，验证“传统CV角点 vs 模型heatmap解码角点”。

## 8. 验收标准

- 训练可稳定收敛，无 shape 或 dtype 错误。
- 单图推理能输出 4 个角点并正确绘制轮廓。
- 分类结果正常输出。
- 相比旧坐标回归版本，角点预测在遮挡/高曝光场景下稳定性提升（通过验证集对比）。

