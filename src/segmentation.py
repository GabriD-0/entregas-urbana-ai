# src/segmentation.py
import cv2
import numpy as np

def segmentar(img, cell_px=32):
    h, w = img.shape[:2]
    h2, w2 = h - h % cell_px, w - w % cell_px
    img = img[:h2, :w2]

    rows, cols = h2 // cell_px, w2 // cell_px
    walkable = np.zeros((rows, cols), bool)
    custo     = np.full((rows, cols), 999, float)   # 999 = intransitável
    tipo      = np.full((rows, cols), "", object)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for i in range(rows):
        for j in range(cols):
            y0, x0 = i*cell_px, j*cell_px
            tile = hsv[y0:y0+cell_px, x0:x0+cell_px]
            h_mean, s_mean, v_mean = tile.reshape(-1,3).mean(axis=0)

            if s_mean < 10:  # tons de cinza
                if v_mean > 200:
                    walkable[i,j], custo[i,j], tipo[i,j] = True, 1, "rua"
                elif v_mean > 120:
                    walkable[i,j], custo[i,j], tipo[i,j] = True, 0.5, "rodovia"
            elif v_mean > 200 and s_mean < 35:
                tipo[i,j] = "predio"              # bloqueado
            elif v_mean < 180 and s_mean > 60 and h_mean < 90:
                tipo[i,j] = "floresta"            # bloqueado
            elif h_mean < 120 and s_mean > 50:
                tipo[i,j] = "rio"                 # bloqueado
            else:
                tipo[i,j] = "outro"

    return walkable, custo, tipo
