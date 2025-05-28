import json
import time
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Sequence

import rota_mapa as rm
from pathfinder import load_graph, a_star
from control import ControlAgent
from delivery import DeliveryAgent

# Configurações do pipeline
grid_size = 16
src_dir = Path("source")
src_dir.mkdir(parents=True, exist_ok=True)

# Caminhos de arquivos
IMAGE_SRC       = src_dir / "image.png"
IMAGE_LINES     = src_dir / "imgs/image_linhas.png"
GRID_IMG        = src_dir / "imgs/image_grid.png"
GRID_LABELS_IMG = src_dir / "imgs/image_grid_labels.png"
GRAPH_JSON      = src_dir / "imgs/image_graph.json"
ROUTE_IMG       = src_dir / "imgs/image_route.png"

# IDs de início e fim para o A*
START_ID = "14_3"
GOAL_ID  = "2_14"
# Número de passos na simulação
ticks = 100


def desenhar_rota(base_img: np.ndarray, coords: Sequence[tuple[int, int]]) -> np.ndarray:
    """
    Desenha a rota sobre a imagem de grid, conectando centros de cada célula.
    - base_img: imagem BGR carregada pelo cv2
    - coords: sequência de tuplas (linha, coluna) representando o percurso planejado
    Retorna uma cópia da imagem com rota desenhada (apenas início e fim marcados).
    """
    img = base_img.copy()
    h, w = img.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size

    # Computa centros dos tiles do percurso
    centers = [
        (c * cell_w + cell_w // 2, r * cell_h + cell_h // 2)
        for r, c in coords
    ]

    # Desenha linhas vermelhas conectando o percurso
    for (x1, y1), (x2, y2) in zip(centers, centers[1:]):
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Marca apenas início (verde) e fim (vermelho)
    if centers:
        sx, sy = centers[0]
        ex, ey = centers[-1]
        cv2.circle(img, (sx, sy), 6, (0, 255, 0), -1)  # início em verde
        cv2.circle(img, (ex, ey), 6, (0, 0, 255), -1)  # fim em vermelho

    return img


def main():
    # 1) Remoção de fundo e máscara
    print("[1/7] Removendo fundo...")
    resultado, mask = rm.remover_fundo(str(IMAGE_SRC))
    cv2.imwrite(str(IMAGE_LINES), resultado)
    print(f"→ {IMAGE_LINES}")

    # 2) Grid
    print("[2/7] Aplicando grid...")
    img_grid = rm.aplicar_grid(resultado, mask, linhas=grid_size, colunas=grid_size)
    if isinstance(img_grid, np.ndarray):
        cv2.imwrite(str(GRID_IMG), img_grid)
    else:
        Image.fromarray(img_grid).save(str(GRID_IMG))
    print(f"→ {GRID_IMG}")

    # 3) Labels
    print("[3/7] Anotando tiles...")
    img_labels = rm.anotar_tiles(img_grid, linhas=grid_size, colunas=grid_size)
    if isinstance(img_labels, np.ndarray):
        cv2.imwrite(str(GRID_LABELS_IMG), img_labels)
    else:
        img_labels.save(str(GRID_LABELS_IMG))
    print(f"→ {GRID_LABELS_IMG}")

    # 4) Grafo
    print("[4/7] Construindo grafo...")
    grafo = rm.construir_grafo(mask, linhas=grid_size, colunas=grid_size)
    GRAPH_JSON.write_text(json.dumps(grafo, indent=2), encoding="utf-8")
    print(f"→ {GRAPH_JSON}")

    # 5) A*
    print(f"[5/7] A* {START_ID}→{GOAL_ID}...")
    positions, adj = load_graph(GRAPH_JSON)
    path_ids = a_star(START_ID, GOAL_ID, positions, adj)
    if path_ids is None:
        raise SystemExit("✖ Sem rota viável")
    print(f"✓ {len(path_ids)-1} passos")

    # Converte IDs "r_c" → (r, c)
    path_coords = [tuple(map(int, node.split('_'))) for node in path_ids]

    # 6) Simulação (histórico opcional)
    print(f"[6/7] Simulando {ticks} ticks...")
    ctrl = ControlAgent(rows=grid_size, cols=grid_size, ttl_alert=4, max_alerts=3, traffic_penalty=3)
    agent = DeliveryAgent("van-01", start_id=START_ID, goal_id=GOAL_ID, graph_json=GRAPH_JSON, control=ctrl)
    ctrl.register(agent)
    start = time.perf_counter()
    for _ in range(ticks): ctrl.step()
    print(f"Simulação em {time.perf_counter()-start:.2f}s")

    # 7) Rota
    print("[7/7] Desenhando rota...")
    base = cv2.imread(str(GRID_IMG))
    img_route = desenhar_rota(base, path_coords) # type: ignore
    cv2.imwrite(str(ROUTE_IMG), img_route)
    print(f"→ {ROUTE_IMG}")

if __name__ == "__main__":
    main()