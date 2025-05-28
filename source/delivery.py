from __future__ import annotations
from pathlib import Path
from typing import List, Set, Tuple, Dict
from pathfinder import a_star, load_graph          # reaproveita código!
from control import ControlAgent

NodeId = str
Coord  = Tuple[int, int]

class DeliveryAgent:
    def __init__(self,
                 agent_id: str,
                 start_id: NodeId,
                 goal_id:  NodeId,
                 graph_json: str | Path,
                 control: ControlAgent) -> None:
        self.id        = agent_id
        self.pos_id    = start_id
        self.goal_id   = goal_id
        self.control   = control
        self.pos_table, self.adj = load_graph(graph_json)

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
            print(f"[{self.id}] -> {self.pos_id}")

    # utilidades internas
    def _plan_route(self) -> None:
        """Compute shortest path with dynamic traffic penalty."""
        def cost(a: NodeId, b: NodeId) -> int:
            # custo base 1 + penalidade de tráfego
            return 1 + self.control.get_penalty(self._coord(b))

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
                    f = tentative + self._manhattan(nxt, self.goal_id)
                    heapq.heappush(open_heap, (f, nxt))
        # sem rota
        self.path = []

    # helpers 
    def _coord(self, node_id: NodeId) -> Coord:
        return self.pos_table[node_id]

    def _manhattan(self, a: NodeId, b: NodeId) -> int:
        r1, c1 = self._coord(a)
        r2, c2 = self._coord(b)
        return abs(r1 - r2) + abs(c1 - c2)

    def _reconstruct(self, came, cur):
        p = [cur]
        while cur in came:
            cur = came[cur]
            p.append(cur)
        p.reverse()
        return p
