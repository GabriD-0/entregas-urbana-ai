# src/segmentation.py
import cv2
import numpy as np

def segmentar(img_bgr, cell_px=32):
    """
    Converte a imagem do mapa em três matrizes:
    • walkable[linha][col]  → bool
    • custo[linha][col]     → float
    • tipo[linha][col]      → string (opcional, p/ debug)
    """
    h, w = img_bgr.shape[:2]

    # 1) ajusta para múltiplo de cell_px
    h2, w2 = h - h % cell_px, w - w % cell_px
    img_bgr = img_bgr[:h2, :w2]

    rows, cols = h2 // cell_px, w2 // cell_px

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    walkable = np.zeros((rows, cols), dtype=bool)
    custo    = np.full((rows, cols), 999.0, dtype=float)
    tipo     = np.full((rows, cols), "",   dtype=object)

    for i in range(rows):
        for j in range(cols):
            y0, x0 = i * cell_px, j * cell_px
            tile   = hsv[y0:y0 + cell_px, x0:x0 + cell_px]

            # Média HSV do tile
            h_mean, s_mean, v_mean = tile.reshape(-1, 3).mean(axis=0)

            # --- regras de cor ---
            # 1) rodovia amarela
            if 15 <= h_mean <= 40 and s_mean > 80 and v_mean > 130:
                walkable[i, j] = True
                custo[i, j]    = 0.5
                tipo[i, j]     = "rodovia"

            # 2) rua branca
            elif s_mean < 40 and v_mean > 200:
                walkable[i, j] = True
                custo[i, j]    = 1.0
                tipo[i, j]     = "rua"

            # 3) tudo o resto é bloqueado
            else:
                # custo já está 999
                tipo[i, j] = "bloqueado"

    return walkable, custo, tipo
