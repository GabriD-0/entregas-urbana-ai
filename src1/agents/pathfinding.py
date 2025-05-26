from heapq import heappush, heappop
from typing import Dict, List, Tuple

Cell = Tuple[int, int]        # (row, col)
Grid = List[List[str]]

def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(grid: Grid, start: Cell, goal: Cell) -> List[Cell]:
    """A* sobre grade 4×4 ('0' = livre, outros = bloqueio)."""
    N = len(grid)
    open_set = []
    g_cost: Dict[Cell, int] = {start: 0}
    came: Dict[Cell, Cell] = {}
    heappush(open_set, (manhattan(start, goal), start))
    while open_set:
        _, cur = heappop(open_set)
        if cur == goal:
            # reconstruir
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for d in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (cur[0]+d[0], cur[1]+d[1])
            if not (0 <= nxt[0] < N and 0 <= nxt[1] < N):
                continue
            if grid[nxt[0]][nxt[1]] != '0':   # obstáculo
                continue
            tentative = g_cost[cur] + 1
            if tentative < g_cost.get(nxt, 1e9):
                came[nxt] = cur
                g_cost[nxt] = tentative
                f = tentative + manhattan(nxt, goal)
                heappush(open_set, (f, nxt))
    return []           # sem caminho
