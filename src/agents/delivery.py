from __future__ import annotations
from heapq import heappush, heappop
from typing import Iterable, Tuple, List, Dict
from agents.control import ControlProxy

"""
Agente de Entrega: navega em um grid 4×4 usando A* e responde a alertas
de tráfego vindos do Agente de Controle.
"""

Coord = Tuple[int, int]                 # (linha, coluna)

class DeliveryAgent:
    def __init__(
        self,
        agent_id: str,
        start: Coord,
        goal: Coord,
        base_map: List[str],             # 16 tokens ['0','X',...]
        controller: "ControlProxy",      # qualquer objeto com .send()/on()
    ):
        self.id = agent_id
        self.pos = start
        self.goal = goal
        self.controller = controller
        self.map = base_map[:]           # cópia mutável
        self.size = 4
        self.path: List[Coord] = []
        self.plan_route()

        # registra-se no controle para receber alertas
        self.controller.subscribe(self.id, self.on_alert)


    # Planejamento 
    def plan_route(self) -> None:
        self.path = self._a_star(self.pos, self.goal)

    def _a_star(self, start: Coord, goal: Coord) -> List[Coord]:
        def h(p: Coord) -> int:          # heurística Manhattan
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        def neighbors(p: Coord) -> Iterable[Coord]:
            r, c = p
            for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    idx = nr*4 + nc
                    if self.map[idx] != "X":         # bloqueio fixo
                        yield (nr, nc)

        open_set: List[Tuple[int, Coord]] = []
        heappush(open_set, (h(start), start))
        came_from: Dict[Coord, Coord] = {}
        g: Dict[Coord, int] = {start: 0}

        while open_set:
            _, current = heappop(open_set)
            if current == goal:
                return self._reconstruct(came_from, current)

            for nxt in neighbors(current):
                # custo base 1; +3 se célula tem tráfego "T"
                idx = nxt[0]*4 + nxt[1]
                step = 1 + (3 if self.map[idx] == "T" else 0)
                new_g = g[current] + step
                if new_g < g.get(nxt, 1e9):
                    came_from[nxt] = current
                    g[nxt] = new_g
                    heappush(open_set, (new_g + h(nxt), nxt))

        return []  # sem rota

    @staticmethod
    def _reconstruct(came: Dict[Coord, Coord], cur: Coord) -> List[Coord]:
        path = [cur]
        while cur in came:
            cur = came[cur]
            path.append(cur)
        path.reverse()
        return path


    # Execução
    def step(self) -> None:
        """Move 1 célula, reporta progresso e replana se necessário."""
        if len(self.path) > 1:
            self.path.pop(0)            # posição atual
            self.pos = self.path[0]

        self.controller.send_progress(self.id, self.pos)

        if self.pos == self.goal:
            self.controller.send_event(self.id, "DELIVERED")

    # Reação a alertas
    def on_alert(self, coords: Iterable[Coord]) -> None:
        """Recebe células com tráfego e atualiza o mapa local."""
        for r, c in coords:
            idx = r*4 + c
            self.map[idx] = "T"
        self.plan_route()
