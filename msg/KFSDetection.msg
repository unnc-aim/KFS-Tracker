# KFS 单帧关键点检测结果（定位 + 分类 + 计时 + 可选深度）
# 默认发布到 /kfs_tracker/detection
std_msgs/Header header

# 本帧是否得到有效目标
bool valid

# 模型类型，用于下游解释 corners / bbox 字段
#   "heatmap" -> corners[4] 有效；bbox_min/bbox_max 为角点的外接框
#   "bbox"    -> bbox_min/bbox_max 为目标框；corners 留 0
string model_type

# heatmap 模式 4 个角点（原图像素坐标），顺序：TL, BL, BR, TR
# 注：Humble 无 Point2D，使用 Point（z 留 0）
geometry_msgs/Point[4] corners

# 目标外接框（像素）：
#   bbox 模式为目标检测框；heatmap 模式为角点外接框（便于下游统一处理）
geometry_msgs/Point bbox_min
geometry_msgs/Point bbox_max

# 分类结果（ImageClassifierInfer / tracker_classifier.pt）
bool class_valid
int32 class_id        # -1 = 无效
string class_name
float32 class_score   # 0.0 ~ 1.0

# 定位 / 角点模型推理耗时（不含分类模型）
float32 infer_ms

# ---- 筛选指标（heatmap 模式，用于下游判断结果质量）----
# 热力图峰值置信度（4 角点最小值，sigmoid 后 0~1）；低于 ~0.3 通常无目标
float32 min_confidence
# 四边形最大边长 / 最小边长；过大说明角点不合理（某个角点飞了）
float32 aspect_ratio

# 目标中心 RGB-D 距离（米）；仅当输入带对齐深度图（16UC1 mm）时填写
# -1.0 = 未提供深度
float32 center_distance_m
