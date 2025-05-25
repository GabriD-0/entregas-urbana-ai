# visual_px.py
from typing import List, Tuple, Optional
from pathlib import Path
from PIL import Image, ImageDraw

Coord = Tuple[int, int]

def _grid2pix(
    coords: List[Coord],
    scale_x: float,
    scale_y: float,
    radius: int,
    draw: ImageDraw.ImageDraw,
    color: Tuple[int, int, int],
):
    pts = [(x*scale_x + scale_x/2, y*scale_y + scale_y/2) for y, x in coords]
    draw.line(pts, width=3, fill=color)
    for cx, cy in pts:
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=color)

def draw_path_px(
    base_img: Image.Image,
    path1: List[Coord],
    out_path: str | Path,
    grid_shape: Tuple[int, int],          #  ← novo
    radius: int = 2,
    path2: Optional[List[Coord]] = None,
) -> None:
    img      = base_img.convert("RGB")
    draw     = ImageDraw.Draw(img)
    grid_H, grid_W = grid_shape
    scale_x  = img.width  / grid_W
    scale_y  = img.height / grid_H

    _grid2pix(path1, scale_x, scale_y, radius, draw, (255, 0, 0))   # vermelho
    if path2:
        _grid2pix(path2, scale_x, scale_y, radius, draw, (0, 0, 255))  # azul

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print("🗺️  Trajeto salvo em:", out_path)
