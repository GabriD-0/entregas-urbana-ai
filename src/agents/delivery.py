from __future__ import annotations
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Iterable

import numpy as np
from agents.control import ControlProxy   # reaproveita seus prints/pub-sub

Coord = Tuple[int, int]    # (y, x)

class PixelDeliveryAgent:
    def __init__(
        self,
        agent_id: str,
        start: Coord,
        goal: Coord,
        cost_grid: np.ndarray,           # matriz de custos (int16)
        controller: ControlProxy,
    ):
        self.id = agent_id
        self.pos = start
        self.goal = goal
        self.costs = cost_grid.copy()
        self.H, self.W = self.costs.shape
        self.ctrl = controller

        self.path: List[Coord] = []
        self.history: List[Coord] = [start]
        self.plan_route()

    # -------------------------- Planejamento ---------------------------- #
    def plan_route(self) -> None:
        self.path = self._a_star(self.pos, self.goal)

    def _a_star(self, start: Coord, goal: Coord) -> List[Coord]:
        """
        A* sobre a grade de custos:
        · conectividade 4-direções
        · heuristic = distância Manhattan
        · g(n)      = soma de custos inteiros (sem limite de uint8)
        """
        def h(p: Coord) -> int:
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        H, W   = self.H, self.W
        costs  = self.costs        # alias local p/ performance
        g: Dict[Coord, int]    = {start: 0}
        came: Dict[Coord, Coord] = {}

        open_heap: List[Tuple[int, Coord]] = []
        heappush(open_heap, (h(start), start))

        while open_heap:
            _, cur = heappop(open_heap)
            if cur == goal:                     # caminho encontrado
                return self._reconstruct(came, cur)

            y, x = cur
            # vizinhos 4-direções
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if not (0 <= ny < H and 0 <= nx < W):
                    continue                    # fora da grade
                step_cost = int(costs[ny, nx])
                if step_cost == 0:
                    continue                    # obstáculo

                nxt = (ny, nx)
                tentative = g[cur] + step_cost
                if tentative < g.get(nxt, 1_000_000_000):
                    g[nxt]   = tentative
                    came[nxt] = cur
                    heappush(open_heap, (tentative + h(nxt), nxt))

        return []

    @staticmethod
    def _reconstruct(came: Dict[Coord, Coord], cur: Coord) -> List[Coord]:
        path = [cur]
        while cur in came:
            cur = came[cur]
            path.append(cur)
        path.reverse()
        return path

    # ----------------------------- Execução ---------------------------- #
    def step(self) -> None:
        if len(self.path) > 1:
            self.path.pop(0)
            self.pos = self.path[0]
            self.history.append(self.pos)

        self.ctrl.send_progress(self.id, self.pos)
        if self.pos == self.goal:
            self.ctrl.send_event(self.id, "DELIVERED")

    # opcional: responder a alertas (marcar novas células lentas, etc.)
    def on_alert(self, coords: Iterable[Coord]) -> None:
        for y, x in coords:
            self.costs[y, x] = 4          # torna aquelas células custosas (trânsito)
        self.plan_route()
