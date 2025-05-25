from __future__ import annotations
import random
from typing import Dict, Callable, List, Tuple, Iterable

"""
Agente de Controle: mantém visão global, gera alertas de tráfego
e envia-os aos Agentes de Entrega registrados.
"""

Coord = Tuple[int, int]

class ControlAgent:
    def __init__(self, base_map: List[str]):
        self.base_map = base_map[:]          # tokens
        self.subscribers: Dict[str, Callable[[Iterable[Coord]], None]] = {}
        self.alert_history: List[List[Coord]] = []

    # canal básico de pub/sub 
    def subscribe(self, agent_id: str, callback: Callable[[Iterable[Coord]], None]):
        self.subscribers[agent_id] = callback

    def _publish(self, cells: List[Coord]) -> None:
        for cb in self.subscribers.values():
            cb(cells)

    # API simulada chamada pelos agentes 
    def send_progress(self, agent_id: str, pos: Coord):
        print(f"[CTRL] {agent_id} -> pos {pos}")

    def send_event(self, agent_id: str, event: str):
        print(f"[CTRL] {agent_id} EVENT {event}")

    # Lógica de tráfego
    def tick(self):
        """Um 'ciclo' de simulação; decide se cria novos alertas."""
        if random.random() < 0.4:   # ~40 % de chance de novo alerta
            cells = self._select_traffic_cells()
            self.alert_history.append(cells)
            print(f"[CTRL] ALERT {cells}")
            self._publish(cells)

    def _select_traffic_cells(self) -> List[Coord]:
        """Escolhe 1-3 células 'livres' para marcar como tráfego."""
        libres = [divmod(i, 4) for i, t in enumerate(self.base_map) if t == "0"]
        random.shuffle(libres)
        return libres[: random.randint(1, 3)]


# Pequeno “proxy” para evitar import circular em delivery.py
class ControlProxy:
    """Interface mínima exposta ao DeliveryAgent (injeção de depend.)"""
    def __init__(self, core: ControlAgent):        # delega para core
        self.core = core

    def subscribe(self, agent_id, callback):
        self.core.subscribe(agent_id, callback)

    # wrappers
    def send_progress(self, agent_id, pos):
        self.core.send_progress(agent_id, pos)

    def send_event(self, agent_id, event):
        self.core.send_event(agent_id, event)
