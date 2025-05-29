import json
import time
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Sequence
import rota_mapa as rm
from pathfinder import load_graph, a_star, dijkstra
from control import ControlAgent
from delivery import DeliveryAgent

# Configurações do pipeline
grid_size = 16
src_dir = Path("source")
src_dir.mkdir(parents=True, exist_ok=True)

# Caminhos de arquivos
IMAGE_SRC       = src_dir / "image.png"
IMAGE_LINES     = src_dir / "imgs/1_image_linhas.png"
GRID_IMG        = src_dir / "imgs/2_image_grid.png"
GRID_LABELS_IMG = src_dir / "imgs/3_image_grid_labels.png"
GRAPH_JSON      = src_dir / "imgs/image_graph.json"
ROUTE_IMG       = src_dir / "imgs/4_image_route_manhattan.png"
IMG_ROUTE_EUC   = src_dir / "imgs/5_image_route_euclid.png"
IMG_ROUTE_DIJ   = src_dir / "imgs/6_image_route_dijk.png"


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
    positions, adj, is_road = load_graph(GRAPH_JSON)

    # 5a) Rota Manhattan (como antes)
    path_ids_man = a_star(START_ID, GOAL_ID, positions, adj, heuristic="manhattan")

    # 5b) Rota Euclidiana
    path_ids_euc = a_star(START_ID, GOAL_ID, positions, adj, heuristic="euclidean")

    # 5c) Dijkstra
    path_ids_dij = dijkstra(START_ID, GOAL_ID, adj)


    if path_ids_man is None or path_ids_euc is None or path_ids_dij is None:
        raise SystemExit("✖ Sem rota viável")


    # Converte IDs "r_c" → (r, c)
    path_coords_man = [tuple(map(int, node.split('_'))) for node in path_ids_man]
    path_coords_euc = [tuple(map(int, node.split('_'))) for node in path_ids_euc]
    path_coords_dij = [tuple(map(int, node.split('_'))) for node in path_ids_dij]
    


    # 6) Simulação (histórico opcional)
    print(f"[6/7] Simulando {ticks} ticks...")
    ctrl = ControlAgent(rows=grid_size, cols=grid_size, ttl_alert=4, max_alerts=3, traffic_penalty=3)

    PERM_BLOCKS = {(13, 3), (8, 9), (7, 3)}

    agent1     = DeliveryAgent("van-01", heuristic="Manhattan", start_id=START_ID, goal_id=GOAL_ID, graph_json=GRAPH_JSON, control=ctrl, permanent_blocks=PERM_BLOCKS)
    agent2     = DeliveryAgent("van-02", heuristic="euclidean", start_id=START_ID, goal_id=GOAL_ID, graph_json=GRAPH_JSON, control=ctrl, permanent_blocks=PERM_BLOCKS)
    agent_dijk = DeliveryAgent("van-dijk", strategy="dijkstra", start_id=START_ID, goal_id=GOAL_ID, graph_json=GRAPH_JSON, control=ctrl, permanent_blocks=PERM_BLOCKS)

    ctrl.register(agent1)
    ctrl.register(agent2)
    ctrl.register(agent_dijk)


    # DEBUG: veja quem está no controle
    print("Agentes registrados →", [ag.id for ag in ctrl._agents])
    
    start = time.perf_counter()
    for _ in range(ticks): ctrl.step()
    print(f"Simulação em {time.perf_counter()-start:.2f}s")

    

    # 7) Rotas
    print("[7/7] Desenhando rota...")
    base = cv2.imread(str(GRID_IMG))
    # — Manhattan em vermelho (default)
    img_route_man = desenhar_rota(base, path_coords_man)# type: ignore
    cv2.imwrite(str(ROUTE_IMG), img_route_man)

    # 7.5) Desenha o caminho real que cada agente percorreu (histórico)
    coords_man = [tuple(map(int, nid.split("_"))) for nid in agent1.history]
    coords_euc = [tuple(map(int, nid.split("_"))) for nid in agent2.history]
    coords_dij = [tuple(map(int, nid.split("_"))) for nid in agent_dijk.history]

    img_hist_man = desenhar_rota(base, coords_man)
    cv2.imwrite(str(src_dir / "imgs/7_rota_real_manhattan.png"), img_hist_man)

    img_hist_euc = desenhar_rota(base, coords_euc)
    cv2.imwrite(str(src_dir / "imgs/8_rota_real_euclidiana.png"), img_hist_euc)

    img_hist_dij = desenhar_rota(base, coords_dij)
    cv2.imwrite(str(src_dir / "imgs/9_rota_real_dijkstra.png"), img_hist_dij)

    print("→ rota_real_manhattan.png, rota_real_euclidiana.png, rota_real_dijkstra.png geradas")


    # — Euclidiana em azul
    def desenhar_rota_azul(img_base, coords):
        azul = (255, 0, 0)
        img = img_base.copy()
        h, w = img.shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        centers = [(c*cell_w+cell_w//2, r*cell_h+cell_h//2) for r, c in coords]
        for p, q in zip(centers, centers[1:]):
            cv2.line(img, p, q, azul, 2)
        cv2.circle(img, centers[0], 6, (0,255,0), -1)
        cv2.circle(img, centers[-1],6, azul, -1)
        return img

    def desenhar_rota_cor(img_base, coords, cor_bgr):
        img = img_base.copy()
        h, w = img.shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        centers = [(c*cell_w+cell_w//2, r*cell_h+cell_h//2) for r, c in coords]
        for p, q in zip(centers, centers[1:]):
            cv2.line(img, p, q, cor_bgr, 2)
        cv2.circle(img, centers[0], 6, (0,255,0), -1)
        cv2.circle(img, centers[-1],6, cor_bgr, -1)
        return img
    
    img_route_euc = desenhar_rota_azul(base, path_coords_euc)
    cv2.imwrite(str(IMG_ROUTE_EUC), img_route_euc)

    img_route_dij = desenhar_rota_cor(base, path_coords_dij, (255,255,0))     # ciano
    cv2.imwrite(str(IMG_ROUTE_DIJ), img_route_dij)

    print(f"-> {ROUTE_IMG}  (Manhattan)")
    print(f"-> {IMG_ROUTE_EUC} (Euclidiana)")
    print(f"-> {IMG_ROUTE_DIJ} (Dijkstra)")


    # 8) Mostrar histórico de cada tick lado a lado
    print("\nTick |   Manhattan   |  Euclidiana   |   Dijkstra   ")
    print("-------+---------------+---------------+--------------")

    # o primeiro elemento de history é o start_id (tick 0)
    max_ticks = max(len(a.history) for a in (agent1, agent2, agent_dijk))

    for t in range(max_ticks):
        m = agent1.history[t] if t < len(agent1.history) else "–"
        e = agent2.history[t] if t < len(agent2.history) else "–"
        d = agent_dijk.history[t] if t < len(agent_dijk.history) else "–"
        print(f"{t:4d} | {m:^13} | {e:^13} | {d:^13}")


if __name__ == "__main__":
    main()