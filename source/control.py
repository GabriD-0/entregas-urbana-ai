from __future__ import annotations
import random
from typing import Dict, List, Tuple

Coord = Tuple[int, int] # (row, col)

class ControlAgent:
    """
    Orquestra o ambiente global.
      • Gere/expira alertas de tráfego.
      • Publica essas mudanças a todos os DeliveryAgents registrados.
      • (opcional) Pode redistribuir entregas, coletar métricas etc.
    """
    def __init__(
        self,
        rows: int,
        cols: int,
        ttl_alert: int = 4,        # quantos “ticks” dura cada alerta
        max_alerts: int = 2,       # quantos bloq. simultâneos
        traffic_penalty: int = 3   # custo extra que o DeliveryAgent deve somar
    ) -> None:

        self.rows      = rows
        self.cols      = cols
        self.ttl_alert = ttl_alert
        self.max_alerts= max_alerts
        self.penalty   = traffic_penalty

        # estado interno
        self._traffic: Dict[Coord,int] = {}        # célula -> TTL restante
        self._agents : List = []                 # referências aos agentes inscritos
        self.tick = 0

    # Interface pública
    def register(self, agent) -> None:
        """Associa um DeliveryAgent a este controle."""
        self._agents.append(agent)
        # envia estado de tráfego atual já na inscrição
        agent.on_traffic_update(set(self._traffic))

    def get_penalty(self, cell: Coord) -> int:
        """Quanto custa atravessar `cell` agora."""
        return self.penalty if cell in self._traffic else 0

    # Loop de simulação
    def step(self) -> None:
        """Avança UM passo na simulação."""
        self.tick += 1
        self._decair_alertas()
        self._gerar_novos_alertas()

        # notifica todo mundo de uma só vez (pub-sub simples)
        atual = set(self._traffic)
        for ag in self._agents:
            ag.on_traffic_update(atual)

        # deixa cada agente agir depois da atualização
        for ag in self._agents:
            ag.next_step()

    # Algoritmos internos
    def _decair_alertas(self) -> None:
        """Reduz TTL de cada alerta; remove os expirados."""
        expirar = [cell for cell, ttl in self._traffic.items() if ttl <= 1]
        for cell in expirar:
            del self._traffic[cell]
        for cell in self._traffic:
            self._traffic[cell] -= 1

    def _gerar_novos_alertas(self) -> None:
        """
        Cria aleatoriamente até `max_alerts` células com tráfego.
        Pode usar lógica mais elaborada (sensores, densidade…), se quiser.
        """
        faltam = self.max_alerts - len(self._traffic)
        while faltam > 0:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            cell = (r, c)
            if cell not in self._traffic: # evita duplicar
                self._traffic[cell] = self.ttl_alert
                faltam -= 1
