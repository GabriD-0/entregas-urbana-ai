from __future__ import annotations
from pathlib import Path
from typing import List, Set, Tuple, Dict
import time, heapq

from pathfinder import dijkstra, load_graph
from control import ControlAgent

NodeId = str
Coord  = Tuple[int, int]

class DeliveryAgent:
    def __init__(
        self,
        agent_id: str,
        start_id: NodeId,
        goal_id:  NodeId,
        graph_json: str | Path,
        control: ControlAgent,
        strategy: str = "astar",
        heuristic: str = "manhattan",
        permanent_blocks: set[Coord] | None = None,
    ) -> None:

        # ----------------- estado geral ----------------- #
        self.id        = agent_id
        self.pos_id    = start_id
        self.goal_id   = goal_id
        self.control   = control
        self.strategy  = strategy.lower()
        self.heuristic = heuristic.lower()
        self.permanent_blocks = set(permanent_blocks or [])

        # grafo
        self.pos_table, self.adj, self.is_road = load_graph(graph_json)

        # histórico de posições
        self.history: List[NodeId] = [start_id]

        # ----------------- métricas --------------------- #
        self.replan_count: int = 0
        self.total_planning_time: float = 0.0
        self.initial_plan_time: float | None = None

        # ------------------------------------------------ #
        self.traffic: Set[Coord] = set()
        self.path:   List[NodeId] = []
        self._plan_route()             # rota inicial

    # ------------ callbacks / integração --------------- #
    def on_traffic_update(self, traffic_cells: Set[Coord]) -> None:
        self.traffic = traffic_cells
        self._plan_route()

    def next_step(self) -> None:
        if self.pos_id == self.goal_id:
            return

        # replaneja se necessário
        if len(self.path) <= 1 or self._coord(self.path[1]) in self.traffic:
            self._plan_route()

        # move 1 passo
        if len(self.path) > 1:
            self.pos_id = self.path.pop(1)
            self.history.append(self.pos_id)
            print(f"[{self.id}] -> {self.pos_id}")

    # ---------------- planejamento --------------------- #
    def _plan_route(self) -> None:
        t0 = time.perf_counter()

        # custo dinâmico (tráfego + blocos permanentes)
        def cost(a: NodeId, b: NodeId) -> int:
            r, c = self._coord(b)
            if (r, c) in self.permanent_blocks:
                return 1_000_000_000
            return 1 + self.control.get_penalty((r, c))

        if self.strategy == "dijkstra":
            self.path = dijkstra(self.pos_id, self.goal_id, self.adj, cost_fn=cost) or []
            self._update_metrics(time.perf_counter() - t0)
            return

        # ---------- A* ----------
        open_heap: List[Tuple[int, NodeId]] = [(0, self.pos_id)]
        g: Dict[NodeId, int] = {self.pos_id: 0}
        came: Dict[NodeId, NodeId] = {}

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == self.goal_id:
                self.path = self._reconstruct(came, cur)
                self._update_metrics(time.perf_counter() - t0)
                return

            for nxt in self.adj[cur]:
                tentative = g[cur] + cost(cur, nxt)
                if nxt not in g or tentative < g[nxt]:
                    came[nxt] = cur
                    g[nxt] = tentative
                    h = self._heuristic(nxt, self.goal_id)
                    heapq.heappush(open_heap, (tentative + h, nxt))

        # sem rota
        self.path = []
        self._update_metrics(time.perf_counter() - t0)

    # ----------------- métricas ------------------------ #
    def _update_metrics(self, dt: float) -> None:
        self.replan_count += 1
        self.total_planning_time += dt
        if self.initial_plan_time is None:
            self.initial_plan_time = dt

    # ---------------- heurísticas ---------------------- #
    def _heuristic(self, a: NodeId, b: NodeId) -> float:
        if self.heuristic == "euclidean":
            return self._euclidean(a, b)
        if self.heuristic == "obstacles":
            return self._obstacles(a, b)
        return self._manhattan(a, b)          # default

    def _euclidean(self, a: NodeId, b: NodeId) -> float:
        r1, c1 = self._coord(a); r2, c2 = self._coord(b)
        return ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5

    def _manhattan(self, a: NodeId, b: NodeId) -> int:
        r1, c1 = self._coord(a); r2, c2 = self._coord(b)
        return abs(r1 - r2) + abs(c1 - c2)

    def _obstacles(self, a: NodeId, b: NodeId) -> int:
        r1, c1 = self._coord(a); r2, c2 = self._coord(b)
        rmin, rmax = sorted((r1, r2)); cmin, cmax = sorted((c1, c2))
        return sum(
            1 for r in range(rmin, rmax + 1)
              for c in range(cmin, cmax + 1)
              if not self.is_road.get((r, c), False)
        )

    # ------------------ util --------------------------- #
    def _coord(self, node_id: NodeId) -> Coord:
        return self.pos_table[node_id]

    def _reconstruct(self, came, cur):
        path = [cur]
        while cur in came:
            cur = came[cur]; path.append(cur)
        path.reverse()
        return path
