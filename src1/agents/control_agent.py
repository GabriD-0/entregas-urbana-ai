import random
from typing import List, Tuple

from .delivery_agent import DeliveryAgent
from .gemini_api import summarise_route
from .pathfinding import Cell, Grid

class ControlAgent:
    """Orquestra múltiplos DeliveryAgent (aqui só 2)."""
    def __init__(self, grid: Grid):
        self.grid = grid
        self.agents: List[DeliveryAgent] = []

    # ---------- sorteio origem/destino válidos -------------------------
    def _sample_free(self) -> Cell:
        frees = [(r,c) for r,row in enumerate(self.grid)
                       for c,val in enumerate(row) if val == '0']
        return random.choice(frees)

    def spawn_agents(self, k: int = 2):
        for i in range(k):
            s, g = self._sample_free(), self._sample_free()
            while g == s:
                g = self._sample_free()
            agent = DeliveryAgent(f"van-{i+1}", self.grid, s, g)
            self.agents.append(agent)

    # ---------- relatório Gemini (opcional) ----------------------------
    def report(self):
        for ag in self.agents:
            summarise_route(ag.id, ag.path)
