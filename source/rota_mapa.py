import cv2
import numpy as np
import json
from typing import Tuple, Dict

def remover_fundo(caminho_entrada: str) -> Tuple[np.ndarray, np.ndarray]:
    # 1) Carrega imagem e checa se abriu
    img = cv2.imread(caminho_entrada)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir '{caminho_entrada}'")

    # 2) Converte pra HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3) Máscara para o branco
    lower_white  = np.array([0,  0, 200])
    upper_white  = np.array([180,30,255])
    mask_white   = cv2.inRange(hsv, lower_white, upper_white)

    # 4) Máscara para o amarelo
    lower_yellow = np.array([15,100,100])
    upper_yellow = np.array([35,255,255])
    mask_yellow  = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 5) Combina e limpa ruídos
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # 6) Aplica máscara para extrair apenas as linhas
    resultado = cv2.bitwise_and(img, img, mask=mask)
    return resultado, mask


def aplicar_grid(
    img: np.ndarray,
    mask: np.ndarray,
    linhas: int = 4,
    colunas: int = 4,
    cor_grade: tuple = (255,255,255),
) -> np.ndarray:
    h, w    = mask.shape[:2]
    tile_h  = h // linhas
    tile_w  = w // colunas
    img_grid = img.copy()
    espessura = 1

    # Desenha as linhas da grid
    for c in range(1, colunas):
        x = c * tile_w
        cv2.line(img_grid, (x, 0), (x, h), cor_grade, espessura)
    for r in range(1, linhas):
        y = r * tile_h
        cv2.line(img_grid, (0, y), (w, y), cor_grade, espessura)

    return img_grid


def construir_grafo(
    mask: np.ndarray,
    linhas: int = 16,
    colunas: int = 16,
    limiar: float = 0.05
) -> Dict:
    H, W = mask.shape[:2]
    th, tw = H // linhas, W // colunas

    # 1) Define quais células são via (True/False)
    road = {}
    for r in range(linhas):
        for c in range(colunas):
            bloco = mask[r*th:(r+1)*th, c*tw:(c+1)*tw]
            proporcao = np.count_nonzero(bloco) / (th*tw)
            road[(r, c)] = proporcao > limiar

    # 2) Monta os nós
    nodes = [
        {"id": f"{r}_{c}", "row": r, "col": c, "is_road": int(road[(r,c)])}
        for r in range(linhas) for c in range(colunas)
    ]

    # 3) Verifica arestas entre vizinhos N/S/E/O
    edges = []
    vizinhos = [(-1,0),(1,0),(0,-1),(0,1)]
    for (r, c), ok in road.items():
        if not ok:
            continue
        for dr, dc in vizinhos:
            nr, nc = r+dr, c+dc
            if 0 <= nr < linhas and 0 <= nc < colunas and road[(nr, nc)]:
                # testa se há pixels de rua na borda compartilhada
                if dr != 0:
                    y = (r + (dr>0)) * th
                    f1 = mask[y, c*tw:(c+1)*tw]
                    f2 = mask[y, nc*tw:(nc+1)*tw]
                else:
                    x = (c + (dc>0)) * tw
                    f1 = mask[r*th:(r+1)*th, x]
                    f2 = mask[nr*th:(nr+1)*th, x]
                if np.any(f1>0) and np.any(f2>0):
                    id1, id2 = f"{r}_{c}", f"{nr}_{nc}"
                    if [id2, id1] not in edges:
                        edges.append([id1, id2])

    return {"nodes": nodes, "edges": edges}

# --- NOVA FUNÇÃO -----------------------------------------------------------
def anotar_tiles(img: np.ndarray,
                 linhas: int,
                 colunas: int,
                 cor_texto: Tuple[int,int,int] = (0,255,0),
                 font_scale: float = 0.4,
                 espessura: int = 1) -> np.ndarray:
    """
    Escreve 'r_c' (linha_coluna) no centro de cada célula.
    Retorna uma cópia anotada da imagem.
    """
    h, w = img.shape[:2]
    tile_h, tile_w = h // linhas, w // colunas
    out = img.copy()

    for r in range(linhas):
        for c in range(colunas):
            cx = c*tile_w + tile_w//2
            cy = r*tile_h + tile_h//2
            texto = f"{r}_{c}"
            # Ajuste fino: move o texto um pouco para cima (‐10) para ficar centralizado
            cv2.putText(out, texto, (cx-20, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, cor_texto, espessura, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    # 1) Remove fundo e obtém a máscara
    resultado, mask = remover_fundo("source/image.png")
    cv2.imwrite("source/image_linhas.png", resultado)

    # 2) (Opcional) desenha grid e pontos
    img_grid = aplicar_grid(resultado, mask, linhas=16, colunas=16)
    cv2.imwrite("source/image_grid.png", img_grid)

    # 2b) (opcional) adiciona etiquetas r_c em cada tile
    img_rotulado = anotar_tiles(img_grid, linhas=16, colunas=16)
    cv2.imwrite("source/image_grid_labels.png", img_rotulado)
    print("Imagem de referência com labels em source/image_grid_labels.png")

    # 3) Constrói o grafo e salva em JSON
    grafo = construir_grafo(mask)
    with open("source/image_graph.json", "w") as f:
        json.dump(grafo, f, indent=2)

    print("Processamento concluído. Grafo viário em source/image_graph.json")
