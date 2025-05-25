from __future__ import annotations
from pathlib import Path
from typing import Tuple
from numpy.typing import NDArray          #  ← novo
import numpy as np
from PIL import Image

# ------------------------ parâmetros heurísticos ------------------------ #
# ↓ ajuste estes limiares para a sua paleta de cores
TH_ROAD_MIN = 190          # pixels muito claros → rua (custo 1)
TH_ROAD_MAX = 255

TH_SLOW_MIN = 120          # cinza médio = zona “lenta” (custo 4)
TH_SLOW_MAX = 189

Array = np.ndarray

# Água costuma ter azul alto e vermelho baixo
def _is_water(r: Array, g: Array, b: Array) -> NDArray[np.bool_]:
    return (b > 150) & (g > 150) & (r < 100)

def build_cost_grid(img_path: str | Path, scale: int = 1) -> Tuple[np.ndarray, Image.Image]:
    """
    Converte uma imagem RGB em uma matriz 2-D de custos (int16).

    · scale > 1 → agressiva: funde (scale×scale) pixels num único custo médio,
      reduzindo o tamanho da grade (ótimo p/ desempenho).
    · Retorna (costs, img_resized) onde:
        costs[y, x] = 0   → obstáculo
                     = 1   → via livre
                     = 4   → via lenta
    """
    img = Image.open(img_path).convert("RGB")

    if scale > 1:
        # reduz a resolução para acelerar A*
        w, h = img.size
        img = img.resize((w // scale, h // scale), Image.Resampling.BILINEAR)

    arr = np.asarray(img)        # shape (H, W, 3)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = (0.3 * r + 0.59 * g + 0.11 * b).astype(np.uint8)

    costs = np.full(gray.shape, 0, dtype=np.uint8)   # obstáculo por padrão

    # 1) Água = obstáculo (custo 0)
    mask_water = _is_water(r, g, b)
    costs[mask_water] = 0

    # 2) Ruas claras → custo 1
    mask_road = (gray >= TH_ROAD_MIN) & (gray <= TH_ROAD_MAX) & (~mask_water)
    costs[mask_road] = 1

    # 3) Cinza médio → custo 4 (trânsito/zona lenta)
    mask_slow = (gray >= TH_SLOW_MIN) & (gray <= TH_SLOW_MAX) & (~mask_water)
    costs[mask_slow] = 4

    return costs.astype(np.int16), img   # retorna custos e imagem (redimensionada se aplicável)
