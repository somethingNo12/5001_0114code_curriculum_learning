"""
课程学习地图生成脚本
生成 10 张 bitmap（region0.jpg ~ region9.jpg），障碍物密度随难度递增
每张图对应一个 20×20m 的训练区域，分辨率与原 d6.jpg 一致（302×302 像素）
"""

import random
import math
from PIL import Image, ImageDraw

# ============================================================
# 参数配置
# ============================================================
IMG_SIZE      = 302        # 像素尺寸（与 d6.jpg 一致）
BORDER_WIDTH  = 8          # 外边框宽度（像素）
MIN_OBS_W     = 6          # 障碍物最小宽度（像素）
MAX_OBS_W     = 28         # 障碍物最大宽度（像素）
MIN_OBS_H     = 6          # 障碍物最小高度（像素）
MAX_OBS_H     = 28         # 障碍物最大高度（像素）
MAX_ANGLE_DEG = 30         # 最大旋转角度（度），模拟 d6.jpg 中的斜置障碍物
MARGIN        = BORDER_WIDTH + 4   # 障碍物距边框的最小距离（像素）
MIN_SPACING   = 4          # 障碍物之间的最小间距（像素）
MAX_RETRY     = 500        # 每个障碍物最多重试次数

# 各 Level 的障碍物数量
OBSTACLE_COUNTS = [0, 3, 6, 9, 12, 15, 18, 21, 26, 32]

OUTPUT_DIR = "."           # 输出目录（相对于脚本位置）

# ============================================================
# 辅助函数
# ============================================================

def rotated_rect_bbox(cx, cy, w, h, angle_deg):
    """计算旋转矩形的轴对齐包围盒（用于碰撞检测）"""
    angle_rad = math.radians(angle_deg)
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    bw = int(w * cos_a + h * sin_a)
    bh = int(w * sin_a + h * cos_a)
    return (cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2)


def boxes_overlap(b1, b2, spacing=MIN_SPACING):
    """判断两个 AABB 包围盒是否重叠（含间距）"""
    return not (
        b1[2] + spacing < b2[0] or
        b2[2] + spacing < b1[0] or
        b1[3] + spacing < b2[1] or
        b2[3] + spacing < b1[1]
    )


def draw_rotated_rect(draw, cx, cy, w, h, angle_deg, fill="black"):
    """在 ImageDraw 上绘制旋转矩形（多边形近似）"""
    angle_rad = math.radians(angle_deg)
    hw, hh = w / 2, h / 2
    corners = [
        (-hw, -hh), ( hw, -hh),
        ( hw,  hh), (-hw,  hh),
    ]
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotated = [
        (cx + x * cos_a - y * sin_a,
         cy + x * sin_a + y * cos_a)
        for x, y in corners
    ]
    draw.polygon(rotated, fill=fill)


# ============================================================
# 主生成函数
# ============================================================

def generate_region(level: int, obstacle_count: int, seed: int) -> Image.Image:
    """生成单张区域地图"""
    random.seed(seed)

    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="white")
    draw = ImageDraw.Draw(img)

    # 外边框
    draw.rectangle(
        [0, 0, IMG_SIZE - 1, IMG_SIZE - 1],
        outline="black", width=BORDER_WIDTH
    )

    placed_bboxes = []

    for _ in range(obstacle_count):
        placed = False
        for _retry in range(MAX_RETRY):
            w = random.randint(MIN_OBS_W, MAX_OBS_W)
            h = random.randint(MIN_OBS_H, MAX_OBS_H)
            angle = random.uniform(-MAX_ANGLE_DEG, MAX_ANGLE_DEG)

            half_diag = int(math.sqrt(w * w + h * h) / 2) + 2
            cx = random.randint(MARGIN + half_diag, IMG_SIZE - MARGIN - half_diag)
            cy = random.randint(MARGIN + half_diag, IMG_SIZE - MARGIN - half_diag)

            bbox = rotated_rect_bbox(cx, cy, w, h, angle)

            # 确保包围盒在边框内
            if (bbox[0] < MARGIN or bbox[1] < MARGIN or
                    bbox[2] > IMG_SIZE - MARGIN or bbox[3] > IMG_SIZE - MARGIN):
                continue

            # 确保与已放置障碍物不重叠
            if any(boxes_overlap(bbox, pb) for pb in placed_bboxes):
                continue

            draw_rotated_rect(draw, cx, cy, w, h, angle)
            placed_bboxes.append(bbox)
            placed = True
            break

        if not placed:
            print(f"  [Level {level}] 第 {len(placed_bboxes)+1} 个障碍物放置失败（空间不足），跳过")

    return img


# ============================================================
# 批量生成
# ============================================================

def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("开始生成课程学习地图...")
    print(f"输出目录: {script_dir}")
    print()

    for level, count in enumerate(OBSTACLE_COUNTS):
        filename = os.path.join(script_dir, f"region{level}.jpg")
        img = generate_region(level=level, obstacle_count=count, seed=42 + level)
        img.save(filename, "JPEG", quality=95)
        print(f"  ✅ region{level}.jpg  — Level {level}，{count} 个静态障碍物")

    print()
    print("全部生成完成！")


if __name__ == "__main__":
    main()
