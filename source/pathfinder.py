from __future__ import annotations
import json
import heapq
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Utilidades de grafo
Coord    = Tuple[int, int]          # (row, col) da célula
NodeId   = str
AdjTable = Dict[NodeId, Set[NodeId]]
PosTable = Dict[NodeId, Coord]

def load_graph(json_path: str | Path) -> Tuple[PosTable, AdjTable]:
    """
    Lê o arquivo JSON gerado pelo seu pipeline e devolve:
      - positions:  id -> (row, col)
      - adj:        id -> set(id vizinhos)
    Considera apenas nós com is_road == 1.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    # Tabela de posições (somente ruas)
    positions: PosTable = {
        n["id"]: (n["row"], n["col"])
        for n in data["nodes"] if n["is_road"] == 1
    }

    # Tabela de adjacência (aresta se ambos os nós são rua)
    adj: AdjTable = {nid: set() for nid in positions}
    for a, b in data["edges"]:
        if a in positions and b in positions:      # garante que ambos são “rua”
            adj[a].add(b)
            adj[b].add(a)
    return positions, adj

# A* (grade – custo uniforme 1 por passo)
def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start: NodeId, goal: NodeId,
           pos: PosTable, adj: AdjTable) -> List[NodeId] | None:
    """
    Algoritmo A*.  Retorna a lista de nós do caminho (start … goal)
    ou None se não existe rota.
    """
    open_heap: List[Tuple[int, NodeId]] = []
    heapq.heappush(open_heap, (0, start))

    g_score: Dict[NodeId, int] = {start: 0}
    came_from: Dict[NodeId, NodeId] = {}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:  # reconstruir caminho
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in adj[current]:
            tentative_g = g_score[current] + 1   # custo uniforme
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan(pos[neighbor], pos[goal])
                heapq.heappush(open_heap, (f, neighbor))
    return None

# Visualização opcional
def draw_path_on_grid(img_path: str | Path,
                      grid_rows: int, grid_cols: int,
                      path: List[NodeId],
                      out_path: str | Path):
    """
    Desenha a rota (lista de NodeIds) por cima da imagem de grade gerada
    anteriormente e salva em `out_path`.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    h, w = img.shape[:2]
    tile_h, tile_w = h // grid_rows, w // grid_cols

    # converte NodeId “r_c” → centro em pixels
    def center(id_: NodeId) -> Tuple[int, int]:
        r, c = map(int, id_.split("_"))
        cx = c * tile_w + tile_w // 2
        cy = r * tile_h + tile_h // 2
        return cx, cy

    # desenha linhas entre centros consecutivos
    for a, b in zip(path, path[1:]):
        cv2.line(img, center(a), center(b), (0, 0, 255), thickness=2)

    # marca start / goal
    cv2.circle(img, center(path[0]), 5, (0, 255, 0), -1)
    cv2.circle(img, center(path[-1]), 5, (255, 0, 0), -1)

    cv2.imwrite(str(out_path), img)
    print(f"Rota desenhada salva em {out_path}")

