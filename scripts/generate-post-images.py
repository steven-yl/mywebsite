#!/usr/bin/env python3
"""
为每篇文章生成抽象主题图：根据 title/tags 选择视觉风格（flow/diffusion/consistency 等），
输出到 static/images/posts/ 与 content/posts/images/，并更新 front matter。
依赖: pip install Pillow
"""
import os
import re
import math
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    print("请先安装 Pillow: pip install Pillow")
    exit(1)

ROOT = Path(__file__).resolve().parent.parent
POSTS_DIR = ROOT / "content" / "posts"
OUT_DIR = ROOT / "static" / "images" / "posts"
CONTENT_IMAGES = ROOT / "content" / "posts" / "images"
W, H = 1200, 630


def slug_from_path(md_path: Path) -> str:
    stem = md_path.stem
    s = re.sub(r"[\s&()（）]+", "-", stem)
    s = re.sub(r"[^\w\-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-").lower()
    return s or "post"


def get_front_matter(md_path: Path) -> tuple[str, str]:
    """返回 (title, tags_str)。"""
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    title, tags = "", ""
    in_fm = False
    for line in text.splitlines():
        if line.strip() == "---":
            in_fm = not in_fm
            if not in_fm:
                break
            continue
        if not in_fm:
            continue
        if line.strip().startswith("title:"):
            title = line.split("title:", 1)[1].strip().strip('"').strip("'")[:80]
        if line.strip().startswith("tags:"):
            tags = line.split("tags:", 1)[1].strip()
    return title, tags


def theme_from_title_tags(title: str, tags: str, slug: str) -> str:
    """根据标题和标签返回主题风格。"""
    t = (title + " " + tags + " " + slug).lower()
    if "flow matching" in t or "flow matching" in title.lower() or "matching" in t:
        return "flow"
    if "ddpm" in t or "前向" in t or "反向" in t or "denoising" in t:
        return "ddpm"
    if "diffusion" in t and "principle" in t:
        return "diffusion"
    if "consistency" in t:
        return "consistency"
    if "sde" in t or "ode" in t or "离散" in t or "连续" in t:
        return "sde"
    if "elbo" in t:
        return "elbo"
    if "meanflow" in t or "mean flow" in t:
        return "meanflow"
    if "how" in t or "doit" in t or "指南" in t or "example" in t or "解析" in t:
        return "doc"
    return "default"


def seed_from_slug(slug: str) -> int:
    """从 slug 得到确定性整数种子，保证每篇文章不同。"""
    h = 0
    for c in slug:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h


def v(seed: int, i: int, lo: float, hi: float) -> float:
    """确定性变化：同一 seed 同一 i 得到同一值，在 [lo, hi] 之间。"""
    x = ((seed + 1) * (i + 1) * 2654435761) & 0xFFFFFFFF
    return lo + (x / 0xFFFFFFFF) * (hi - lo)


def draw_flow(img: Image.Image, seed: int) -> None:
    """流动曲线：每条曲线数量、相位、颜色随 seed 变化。"""
    draw = ImageDraw.Draw(img)
    n1 = 4 + int(v(seed, 0, 0, 3))
    n2 = 2 + int(v(seed, 1, 0, 3))
    phase1 = v(seed, 2, 0, math.pi * 2)
    phase2 = v(seed, 3, 0, math.pi * 2)
    step = 35 + int(v(seed, 4, 0, 15))
    for i in range(n1):
        y0 = 80 + i * (H // (n1 + 1)) + int(v(seed, 10 + i, -30, 30))
        pts = []
        for x in range(0, W + 1, step):
            y = y0 + 50 * math.sin(x / (160 + v(seed, 5, 0, 40)) + i * 0.7 + phase1) + 25 * math.sin(x / 80 + phase2)
            pts.append((x, y))
        if len(pts) >= 2:
            r = int(70 + v(seed, 20 + i, 0, 50))
            g = int(90 + v(seed, 21 + i, 0, 40))
            b = int(160 + v(seed, 22 + i, 0, 50))
            draw.line(pts, fill=(r, g, b), width=10 + int(v(seed, 30 + i, 0, 6)))
    for i in range(n2):
        y0 = 120 + (i + 1) * (H / (n2 + 2)) + int(v(seed, 40 + i, -25, 25))
        pts = [(x, y0 + 30 * math.sin(x / (180 + v(seed, 50 + i, 0, 40)))) for x in range(0, W + 1, 25)]
        draw.line(pts, fill=(100 + int(v(seed, 60 + i, 0, 40)), 70, 180), width=6 + int(v(seed, 70 + i, 0, 4)))


def draw_ddpm(img: Image.Image, seed: int) -> None:
    """扩散感：同心圆间距、颗粒数量与分布、颜色随 seed 变化。"""
    draw = ImageDraw.Draw(img)
    cx, cy = W // 2 + int(v(seed, 0, -80, 80)), H // 2 + int(v(seed, 1, -50, 50))
    step = 28 + int(v(seed, 2, 0, 20))
    for j, r in enumerate(range(20, 400, step)):
        dr = int(200 + v(seed, 10 + j, 0, 60))
        dg = int(130 + v(seed, 20 + j, 0, 50))
        db = int(80 + v(seed, 30 + j, 0, 40))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(dr, dg, db), width=1 + int(v(seed, 40 + j, 0, 2)))
    n_pts = 150 + int(v(seed, 50, 0, 120))
    for _ in range(n_pts):
        a = math.pi * 2 * ((_ + int(v(seed, 51, 0, 1000))) / max(n_pts, 1))
        r = 60 + (_ % 140) + int(v(seed, 52 + _, 0, 40))
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        size = 2 + int(v(seed, 100 + _, 0, 4))
        draw.ellipse([x - size, y - size, x + size, y + size], fill=(230 + int(v(seed, 200 + _, 0, 20)), 170, 90))


def draw_diffusion(img: Image.Image, seed: int) -> None:
    """扩散原理：射线数量、角度偏移、同心圆半径与颜色随 seed 变化。"""
    draw = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2
    n_rays = 18 + int(v(seed, 0, 0, 14))
    for i in range(n_rays):
        a = (i / n_rays) * math.pi * 2 + v(seed, 1 + i, 0, 0.2)
        length = 600 + v(seed, 2, 0, 150)
        x2, y2 = cx + length * math.cos(a), cy + length * math.sin(a)
        draw.line([(cx, cy), (x2, y2)], fill=(150 + int(v(seed, 3 + i, 0, 30)), 110, 190), width=1 + int(v(seed, 4 + i, 0, 2)))
    radii = [120 + v(seed, 10, 0, 50), 250 + v(seed, 11, 0, 60), 380 + v(seed, 12, 0, 50)]
    for j, r in enumerate(radii):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(90 + int(v(seed, 20 + j, 0, 25)), 75, 140), width=1)


def draw_consistency(img: Image.Image, seed: int) -> None:
    """一致性：线条数量、间距、倾斜、同心圆半径与颜色随 seed 变化。"""
    draw = ImageDraw.Draw(img)
    n_lines = 10 + int(v(seed, 0, 0, 6))
    spacing = 85 + int(v(seed, 1, 0, 30))
    tilt = 40 + int(v(seed, 2, 0, 50))
    for i in range(n_lines):
        x = 60 + i * spacing + int(v(seed, 10 + i, -20, 20))
        draw.line([(x, 0), (x + tilt, H)], fill=(65 + int(v(seed, 20 + i, 0, 25)), 125, 175), width=4 + int(v(seed, 30 + i, 0, 4)))
    for j, r in enumerate(range(70, 300, 35 + int(v(seed, 40, 0, 15)))):
        draw.ellipse([W//2 - r, H//2 - r, W//2 + r, H//2 + r], outline=(85 + int(v(seed, 50 + j, 0, 25)), 145, 195), width=2)


def draw_sde(img: Image.Image, seed: int) -> None:
    """SDE/ODE：行数列数、曲线相位与振幅、颜色随 seed 变化。"""
    draw = ImageDraw.Draw(img)
    n_row = 5 + int(v(seed, 0, 0, 3))
    n_col = 6 + int(v(seed, 1, 0, 4))
    for row in range(n_row):
        y0 = 60 + row * (H / (n_row + 1)) + int(v(seed, 10 + row, -20, 20))
        for col in range(n_col):
            x0 = 80 + col * (W / (n_col + 1)) + int(v(seed, 20 + col, -30, 30))
            pts = []
            amp1 = 12 + v(seed, 30 + row * n_col + col, 0, 15)
            amp2 = 15 + v(seed, 40 + row * n_col + col, 0, 15)
            for t in range(0, 100, 2):
                x = x0 + t * (1.0 + v(seed, 50, 0, 0.5)) + amp1 * math.sin(t / 7 + v(seed, 51, 0, 2))
                y = y0 + t * 0.2 + amp2 * math.cos(t / 9 + v(seed, 52, 0, 2))
                if 0 <= x < W and 0 <= y < H:
                    pts.append((x, y))
            if len(pts) >= 2:
                draw.line(pts, fill=(50 + int(v(seed, 60 + row, 0, 30)), 150 + int(v(seed, 70 + col, 0, 25)), 130), width=2 + int(v(seed, 80, 0, 2)))


def draw_elbo(img: Image.Image, seed: int) -> None:
    """ELBO：椭圆个数、旋转、大小、颜色随 seed 变化。"""
    draw = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2
    n_ell = 6 + int(v(seed, 0, 0, 5))
    base_angle = v(seed, 1, 0, math.pi * 2)
    for i in range(n_ell):
        a = (i / n_ell) * math.pi * 2 + base_angle + v(seed, 2 + i, 0, 0.3)
        rx = 160 + i * 12 + int(v(seed, 10 + i, 0, 30))
        ry = 100 + i * 10 + int(v(seed, 20 + i, 0, 25))
        ex = cx + (rx * 0.5) * math.cos(a)
        ey = cy + (ry * 0.5) * math.sin(a)
        draw.ellipse([ex - rx * 0.5, ey - ry * 0.5, ex + rx * 0.5, ey + ry * 0.5], outline=(130 + int(v(seed, 30 + i, 0, 30)), 90, 200), width=2 + int(v(seed, 40 + i, 0, 2)))


def draw_meanflow(img: Image.Image, seed: int) -> None:
    """MeanFlow：曲线条数、间距、波形与颜色随 seed 变化。"""
    draw = ImageDraw.Draw(img)
    n_lines = 4 + int(v(seed, 0, 0, 4))
    step_y = (H - 160) // n_lines
    for i in range(n_lines):
        y = 90 + i * step_y + int(v(seed, 10 + i, -25, 25))
        pts = []
        waves = 1.5 + v(seed, 20 + i, 0, 1.5)
        amp = 25 + v(seed, 30 + i, 0, 25)
        for x in range(0, W + 1, 20):
            yy = y + amp * math.sin((x / W) * math.pi * 2 * waves + v(seed, 40 + i, 0, 2))
            pts.append((x, yy))
        draw.line(pts, fill=(45 + int(v(seed, 50 + i, 0, 25)), 110, 195), width=8 + int(v(seed, 60 + i, 0, 6)))
    cy = H // 2 + int(v(seed, 70, -30, 30))
    draw.line([(0, cy), (W, cy)], fill=(90 + int(v(seed, 71, 0, 30)), 150, 250), width=5 + int(v(seed, 72, 0, 3)))


def draw_doc(img: Image.Image, seed: int) -> None:
    """文档：网格间距、中心框位置与大小、颜色随 seed 变化。"""
    draw = ImageDraw.Draw(img)
    grid = 60 + int(v(seed, 0, 0, 40))
    for i in range(0, W, grid):
        draw.line([(i, 0), (i, H)], fill=(95 + int(v(seed, 10 + i // grid, 0, 25)), 105, 125))
    for i in range(0, H, grid):
        draw.line([(0, i), (W, i)], fill=(95 + int(v(seed, 20 + i // grid, 0, 25)), 105, 125))
    box_w = 100 + int(v(seed, 30, 0, 60))
    box_h = 60 + int(v(seed, 31, 0, 50))
    cx = W // 2 + int(v(seed, 32, -80, 80))
    cy = H // 2 + int(v(seed, 33, -60, 60))
    draw.rectangle([cx - box_w, cy - box_h, cx + box_w, cy + box_h], outline=(110 + int(v(seed, 34, 0, 30)), 120, 165), width=3 + int(v(seed, 35, 0, 2)))


def draw_default(img: Image.Image, seed: int) -> None:
    """默认：同心圆间距与数量、点的分布与颜色随 seed 变化。"""
    draw = ImageDraw.Draw(img)
    cx = W // 2 + int(v(seed, 0, -60, 60))
    cy = H // 2 + int(v(seed, 1, -40, 40))
    step_r = 45 + int(v(seed, 2, 0, 25))
    for j, r in enumerate(range(90, 360, step_r)):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(90 + int(v(seed, 10 + j, 0, 35)), 115, 165), width=2 + int(v(seed, 20 + j, 0, 2)))
    n_pts = 25 + int(v(seed, 30, 0, 20))
    for i in range(n_pts):
        x = (i / n_pts) * W * 0.8 + W * 0.1 + v(seed, 40 + i, -30, 30)
        y = cy + 70 * math.sin(x / 90 + v(seed, 50 + i, 0, 2))
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(130 + int(v(seed, 60 + i, 0, 30)), 145, 200))


def make_gradient(img: Image.Image, theme: str, seed: int) -> None:
    """根据主题填充渐变底，颜色随 seed 微调使每张图不同。"""
    draw = ImageDraw.Draw(img)
    palettes = {
        "flow": ((25, 35, 55), (45, 55, 95)),
        "ddpm": ((35, 28, 35), (60, 45, 50)),
        "diffusion": ((30, 25, 45), (55, 45, 75)),
        "consistency": ((28, 40, 55), (45, 65, 90)),
        "sde": ((22, 45, 42), (40, 70, 65)),
        "elbo": ((35, 28, 48), (55, 45, 75)),
        "meanflow": ((20, 40, 65), (35, 70, 110)),
        "doc": ((32, 35, 42), (50, 55, 65)),
        "default": ((28, 35, 50), (48, 58, 82)),
    }
    c1, c2 = palettes.get(theme, palettes["default"])
    shift = (int(v(seed, 0, -8, 8)), int(v(seed, 1, -8, 8)), int(v(seed, 2, -8, 8)))
    c1 = tuple(max(0, min(255, c1[i] + shift[i])) for i in range(3))
    c2 = tuple(max(0, min(255, c2[i] + shift[i])) for i in range(3))
    for y in range(H):
        t = y / H
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))


def make_image(title: str, tags: str, slug: str, out_path: Path) -> None:
    theme = theme_from_title_tags(title, tags, slug)
    seed = seed_from_slug(slug)
    img = Image.new("RGB", (W, H), color=(30, 35, 50))
    make_gradient(img, theme, seed)
    drawers = {
        "flow": draw_flow,
        "ddpm": draw_ddpm,
        "diffusion": draw_diffusion,
        "consistency": draw_consistency,
        "sde": draw_sde,
        "elbo": draw_elbo,
        "meanflow": draw_meanflow,
        "doc": draw_doc,
        "default": draw_default,
    }
    drawers.get(theme, draw_default)(img, seed)
    try:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "WEBP", quality=90)


def update_front_matter(md_path: Path, image_url: str) -> None:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    url = image_url if image_url.startswith("/") else "/" + image_url
    text = re.sub(r'(featuredImage:\s*)"[^"]*"', lambda m: m.group(1) + f'"{url}"', text, count=1)
    text = re.sub(r'(featuredImagePreview:\s*)"[^"]*"', lambda m: m.group(1) + f'"{url}"', text, count=1)
    md_path.write_text(text, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONTENT_IMAGES.mkdir(parents=True, exist_ok=True)
    updated = []
    for md_path in sorted(POSTS_DIR.rglob("*.md")):
        if md_path.name.startswith("_") or "/_" in str(md_path):
            continue
        slug = slug_from_path(md_path)
        if not slug:
            continue
        title, tags = get_front_matter(md_path)
        out_name = f"{slug}.webp"
        out_path = OUT_DIR / out_name
        make_image(title, tags, slug, out_path)
        import shutil
        dest = CONTENT_IMAGES / out_name
        shutil.copy2(out_path, dest)
        image_url = f"/images/posts/{out_name}"
        update_front_matter(md_path, image_url)
        theme = theme_from_title_tags(title, tags, slug)
        updated.append((md_path.name, image_url, theme))
    print(f"已生成 {len(updated)} 张抽象主题图 -> static/images/posts/ 与 content/posts/images/")
    for name, url, th in updated:
        print(f"  [{th}] {name} -> {url}")


if __name__ == "__main__":
    main()
