from __future__ import annotations
from pathfinder import dijkstra
from pathlib import Path
from typing import List, Set, Tuple, Dict
from pathfinder import load_graph
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
        permanent_blocks: set[Coord] | None = None
    ) -> None:

        self.history: List[NodeId] = [start_id]

        self.id        = agent_id
        self.pos_id    = start_id
        self.goal_id   = goal_id
        self.control   = control
        self.strategy  = strategy
        self.heuristic = heuristic
        self.permanent_blocks = set(permanent_blocks or [])
        self.pos_table, self.adj, self.is_road = load_graph(graph_json)

        self.traffic: Set[Coord] = set()   # células bloqueadas
        self.path:   List[NodeId] = []
        self._plan_route()                 # rota inicial

    # callbacks do CONTROL
    def on_traffic_update(self, traffic_cells: Set[Coord]) -> None:
        self.traffic = traffic_cells
        self._plan_route()                 # replaneja com custos atualizados

    # loop principal (chamado pelo ControlAgent.step())
    def next_step(self) -> None:
        if self.pos_id == self.goal_id:    # já chegou
            return

        # se a rota está vazia ou o próximo nó ficou bloqueado,
        # replanejar (robustez extra)
        if len(self.path) <= 1 or \
           self._coord(self.path[1]) in self.traffic:
            self._plan_route()

        # avança para o próximo nó
        if len(self.path) > 1:
            self.pos_id = self.path.pop(1)     # move 1 passo
            self.history.append(self.pos_id)

            print(f"[{self.id}] -> {self.pos_id}")

    # utilidades internas
    def _plan_route(self) -> None:
        """Recalcula self.path conforme self.strategy."""
        def cost(a: NodeId, b: NodeId) -> int:
            r, c = self._coord(b)

            if (r, c) in self.permanent_blocks:
                return 1_000_000_000           
            return 1 + self.control.get_penalty(self._coord(b))

        if self.strategy == "dijkstra":
            self.path = dijkstra(
                self.pos_id,
                self.goal_id,
                self.adj,
                cost_fn=cost) or []
            return

        # A*: mesmo heurística Manhattan, mas custo por aresta = cost()
        open_heap: List[Tuple[int, NodeId]] = []
        import heapq
        heapq.heappush(open_heap, (0, self.pos_id))
        g: Dict[NodeId, int] = {self.pos_id: 0}
        came: Dict[NodeId, NodeId] = {}

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == self.goal_id:
                self.path = self._reconstruct(came, cur)
                return
            for nxt in self.adj[cur]:
                tentative = g[cur] + cost(cur, nxt)
                if nxt not in g or tentative < g[nxt]:
                    came[nxt] = cur
                    g[nxt] = tentative

                    if self.heuristic == "euclidean":
                        h = self._eucliedean(nxt, self.goal_id)
                    elif self.heuristic == "obstacles":
                        h = self._obstacles(nxt, self.goal_id)
                    else:
                        h = self._manhattan(nxt, self.goal_id)
                    f = tentative + h

                    heapq.heappush(open_heap, (f, nxt)) # type: ignore
        # sem rota
        self.path = []

    # helpers 
    def _coord(self, node_id: NodeId) -> Coord:
        return self.pos_table[node_id]

    def _eucliedean(self, a: NodeId, b: NodeId) -> float:
        r1, c1 = self._coord(a)
        r2, c2 = self._coord(b)
        return ((r1 - r2)**2 + (c1 - c2)**2) ** 0.5

    def _manhattan(self, a: NodeId, b: NodeId) -> int:
        r1, c1 = self._coord(a)
        r2, c2 = self._coord(b)
        return abs(r1 - r2) + abs(c1 - c2)

    def _obstacles(self, a: NodeId, b: NodeId) -> int:
        """Número de tiles-obstáculo dentro do retângulo start→goal (admissível)."""
        r1, c1 = self._coord(a)
        r2, c2 = self._coord(b)
        rmin, rmax = sorted((r1, r2))
        cmin, cmax = sorted((c1, c2))

        conta = 0
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if not self.is_road.get((r, c), False):
                    conta += 1
        return conta

    def _reconstruct(self, came, cur):
        p = [cur]
        while cur in came:
            cur = came[cur]
            p.append(cur)
        p.reverse()
        return p
    
    def toggle_obstacle(self, cell: Coord, is_road: bool = False) -> None:
        """
        Marca ou desmarca permanentemente uma célula como rua/obstáculo **somente
        para fins da heurística** (não altera o grafo nem o custo real).
        Use antes de iniciar a simulação.
        """
        self.is_road[cell] = is_road

