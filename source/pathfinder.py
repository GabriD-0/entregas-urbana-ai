from __future__ import annotations
import json
import heapq
import math
import cv2
from pathlib import Path
from typing import Dict, List, Set, Tuple, Callable

# Utilidades de grafo
Coord    = Tuple[int, int] # (row, col) da célula
NodeId   = str
AdjTable = Dict[NodeId, Set[NodeId]]
PosTable = Dict[NodeId, Coord]

def load_graph(json_path: str | Path) -> Tuple[PosTable, AdjTable]:
    # Lê o arquivo JSON gerado pelo seu pipeline e devolve:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    # Tabela de posições (somente ruas)
    positions: PosTable = {
        n["id"]: (n["row"], n["col"])
        for n in data["nodes"] if n["is_road"] == 1
    }

    # Tabela de adjacência (aresta se ambos os nós são rua)
    adj: AdjTable = {nid: set() for nid in positions}
    for a, b in data["edges"]:
        if a in positions and b in positions:
            adj[a].add(b)
            adj[b].add(a)

    is_road = {
        (n["row"], n["col"]): bool(n["is_road"])
        for n in data["nodes"]
    }
    return positions, adj, is_road


# A* (grade – custo uniforme 1 por passo)
def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Euclidiana
def euclidiana(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def obstaculos(a: Coord, b: Coord, is_road: Dict[Coord, bool]) -> int:
    r1, c1 = a
    r2, c2 = b

    rmin, rmax = sorted((r1, r2))
    cmin, cmax = sorted((c1, c2))

    conta = 0

    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            if (r, c) not in is_road or not is_road[(r, c)]:
                conta += 1

    return conta

"""
Algoritmo A*.  Retorna a lista de nós do caminho (start … goal)
ou None se não existe rota.
"""
def a_star(
        start: NodeId,
        goal: NodeId,
        pos: PosTable,
        adj: AdjTable,
        heuristic: str = "manhattan",
        is_road: Dict[Coord, bool] | None = None
    ) -> List[NodeId] | None:

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
            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan(pos[neighbor], pos[goal])

                # Heuristica
                if heuristic == "manhattan":
                    h = manhattan(pos[neighbor], pos[goal])
                elif heuristic == "euclidean":
                    h = euclidiana(pos[neighbor], pos[goal])
                elif heuristic == "obstacles":
                    if is_road is None:
                        raise ValueError("Passe is_road se usar heuristica 'obstacles'")
                    h = obstaculos(pos[neighbor], pos[goal], is_road)
                else:
                    raise ValueError(f"Heurística '{heuristic}' desconhecida")
                f = tentative_g + h

                heapq.heappush(open_heap, (f, neighbor)) # type: ignore
    return None


# Dijkstra = Não heuristico
def dijkstra(
        start: NodeId,
        goal: NodeId,
        adj: AdjTable,
        cost_fn: Callable[[NodeId, NodeId], int] = lambda _a, _b: 1
    ) -> List[NodeId] | None:

    open_heap: List[Tuple[int, NodeId]] = [(0, start)]
    came: Dict[NodeId, NodeId] = {}
    dist: Dict[NodeId, int]   = {start: 0}

    while open_heap:
        g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path

        for nxt in adj[cur]:
            ng = g + cost_fn(cur, nxt) # CUSTO REAL
            if nxt not in dist or ng < dist[nxt]:
                dist[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_heap, (ng, nxt))
    return None


    """
    Desenha a rota (lista de NodeIds) por cima da imagem de grade gerada
    anteriormente e salva em `out_path`.
    """
def draw_path_on_grid(
        img_path: str | Path,
        grid_rows: int, grid_cols: int,
        path: List[NodeId],
        out_path: str | Path
    ):

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

