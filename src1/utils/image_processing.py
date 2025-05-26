from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

Cell = Tuple[int, int]            # (row, col) na grade 4×4
TokenGrid = List[List[str]]       # 4×4 com '0','W','X'

# ---------- 1. Carga & corte -------------------------------------------
def load_square(path: Path) -> Image.Image:
    """Abre a imagem e a transforma num quadrado (preenchendo bordas)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    sq = Image.new("RGB", (side, side), (255, 255, 255))
    sq.paste(img, ((side-w)//2, (side-h)//2))
    return sq

# ---------- 2. Grid 4×4 -------------------------------------------------
def overlay_grid(img: Image.Image, n: int = 4,
                 color=(0, 0, 0), width=3) -> Image.Image:
    """Desenha a grade e devolve uma cópia."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    side = out.size[0]
    step = side // n
    for i in range(1, n):
        # linhas horizontais
        draw.line([(0, i*step), (side, i*step)], fill=color, width=width)
        # verticais
        draw.line([(i*step, 0), (i*step, side)], fill=color, width=width)
    return out

# ---------- 3. Tokenização ---------------------------------------------
def _classify_block(arr: np.ndarray) -> str:
    """
    Recebe (H,W,3) uint8 de um bloco e retorna:
    'W' água  |  'X' obstáculo/verde  |  '0' rua livre
    Critérios simples baseado em cor média.
    """
    r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    # muita componente blue  -> água
    if b > 150 and b > r + 20 and b > g + 20:
        return "W"
    # verdes intensos (floresta/grama) tratamos como obstáculo
    if g > 130 and g > r + 15:
        return "X"
    # caso contrário presumimos via ou área branca -> livre
    return "0"

def tokenize(img: Image.Image, n: int = 4) -> TokenGrid:
    """Devolve matriz n×n de tokens."""
    arr = np.asarray(img)
    side = arr.shape[0]
    step = side // n
    grid = []
    for i in range(n):
        row = []
        for j in range(n):
            block = arr[i*step:(i+1)*step, j*step:(j+1)*step, :]
            row.append(_classify_block(block))
        grid.append(row)
    return grid

# ---------- 4. Utilidades ----------------------------------------------
def cell_center(cell: Cell, img_side: int, n: int = 4) -> Tuple[int, int]:
    """Centro (x,y) em pixels de uma célula (row,col)."""
    step = img_side // n
    y = cell[0]*step + step//2
    x = cell[1]*step + step//2
    return (x, y)
