from __future__ import annotations

import json
import os
import time
import re
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Set, Tuple
import google.generativeai as genai
import google.api_core.exceptions as gexc
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY2 = os.getenv("GROQ_API_KEY2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuração do tabuleiro #
GRID_ROWS = 16
GRID_COLS = 16
STUCK_LIMIT = 10
PING_LIMIT = 4

# Tiles permanentes bloqueados (rios, prédios…)
PERMANENT_BLOCKS: Set[Tuple[int, int]] = {
    (13, 2), (8, 3), (7, 3)  # exemplo
}

# Carregamento das rotas #
_METRICS_PATH = Path("source/json/ticks_routes.json")


"""
Extrai a primeira dupla de inteiros da resposta do LLM.
Aceita formatos: '(14,3)', '14,3', '14_3', 'row 14 col 3', '{"row":14,"col":3}', etc.
"""
def _parse_move(ans: str) -> Tuple[int, int] | None:
    nums = re.findall(r"\d+", ans)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return None

def _load_routes() -> Dict[str, List[str]]:
    """Lê rotas do arquivo de métricas; falha se não existir ou formato errado."""
    if not _METRICS_PATH.exists():
        raise FileNotFoundError(f"Arquivo de rotas não encontrado: {_METRICS_PATH}")
    with _METRICS_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    routes = data.get("routes")
    if not isinstance(routes, dict) or not all(isinstance(v, list) for v in routes.values()):
        raise ValueError(f"Formato inválido em 'routes' dentro de {_METRICS_PATH}")
    print(f"✔  Rotas carregadas de {_METRICS_PATH}")
    return routes  # type: ignore[return-value]


ROUTES: Dict[str, List[str]] = _load_routes()

# Utilidades gerais #
def node_to_coord(node_id: str) -> Tuple[int, int]:
    """Converte "r_c" → (row, col)."""
    r, c = map(int, node_id.split("_"))
    return r, c


def neighbors(r: int, c: int) -> List[Tuple[int, int]]:
    """Vizinhos N, S, L, O dentro dos limites do tabuleiro."""
    cand = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [p for p in cand if 0 <= p[0] < GRID_ROWS and 0 <= p[1] < GRID_COLS]


# Interface de provedores de chat 
class ChatProviderBase(ABC):
    @abstractmethod
    def ask(self, prompt: str) -> str:
        ...


# Google Gemini Provider #
class GeminiProvider(ChatProviderBase):
    def __init__(self, model: str = "gemini-2.0-flash") -> None:

        if not GEMINI_API_KEY:
            raise EnvironmentError("Defina GEMINI_API_KEY nas variáveis de ambiente.")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model)

    def ask(self, prompt: str) -> str:
        while True:
            try:
                resp = self.model.generate_content(prompt, safety_settings={})
                return resp.text.strip()
            except gexc.ResourceExhausted as e:
                # lê o retry_delay sugerido pela API
                delay = getattr(e, "retry_delay", 10)
                print(f"[Gemini] quota — aguardando {delay} s…")
                time.sleep(delay)


# Groq LLM Provider
class GroqProvider(ChatProviderBase):
    def __init__(self, model: str = "deepseek-r1-distill-llama-70b") -> None:

        if not GROQ_API_KEY:
            raise EnvironmentError("Defina GROQ_API_KEY nas variáveis de ambiente.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model

    def ask(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

# Groq LLM Provider #
class GroqProvider2(ChatProviderBase):
    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:

        if not GROQ_API_KEY2:
            raise EnvironmentError("Defina GROQ_API_KEY2 nas variáveis de ambiente.")
        self.client = Groq(api_key=GROQ_API_KEY2)
        self.model = model

    def ask(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

# Agente móvel
class ChatControlledAgent:
    def __init__(
        self,
        agent_id: str,
        route: List[str],
        provider: ChatProviderBase,
    ) -> None:

        self.id = agent_id
        self.route_ids = route[:]
        self.provider = provider
        self.cur_idx = 0
        self.pos = node_to_coord(self.route_ids[0])
        self.finished = False
        self.finish_tick = -1 
        self.incapable = False
        self._hist: list[Tuple[int, int]] = []
        self._ping = 0
        self._last_block: Tuple[int, int] | None = None 

    # Passo único #
    def step(self, current_tick: int) -> None:
        if self.finished:
            return
        
        prev_pos = self.pos # posição antes de tentar mover

        goal = node_to_coord(self.route_ids[-1])
        nbrs = neighbors(*self.pos)
        nbrs_valid = [p for p in nbrs if p not in PERMANENT_BLOCKS]

        prompt = (
            f"Você está controlando o agente {self.id} em um grid {GRID_ROWS}×{GRID_COLS}.\n"
            f"Posição atual: {self.pos}.  Objetivo final: {goal}.\n"
            f"Células permanentes bloqueadas: {sorted(PERMANENT_BLOCKS)}.\n"
            + (
                f"Atenção: a célula {self._last_block} está bloqueada. "
                f"Escolha uma das células LIVRES adjacentes listadas abaixo.\n"
              if self._last_block else ""
            )
            + f"Vizinhos livres imediatos: {nbrs_valid}.\n"
            + "Responda somente com a próxima célula (row,col) onde mover o agente."
        )

        try:
            answer = self.provider.ask(prompt)
        except Exception as exc:
            print(f"[{self.id}] erro no provedor: {exc}. Avançando pela rota ideal.")
            answer = "" # Fallback para rota ideal

        move: Tuple[int, int] | None = None

        # Tenta processar a resposta do LLM
        try:
            # usa sempre o parser genérico
            move = _parse_move(answer)
        except Exception:
            move = None  # parser nunca deve lançar, mas garantimos

        # 1)  tentativa com resposta do LLM 
        if move in PERMANENT_BLOCKS:
            print(f"[{self.id}] {move} é bloqueado — voltando para {prev_pos}")
            self.pos = prev_pos
            self._last_block = move
        elif move in nbrs_valid:
            self.pos = move
        else:
            # fallback: tenta avançar pela rota ideal UM passo
            next_idx = self.cur_idx + 1
            # pula todos os passos bloqueados consecutivamente
            while next_idx < len(self.route_ids):
                cand = node_to_coord(self.route_ids[next_idx])
                if cand in PERMANENT_BLOCKS:
                    # registra e **pula** o tile bloqueado
                    print(f"[{self.id}] rota ideal bateu em {cand} — pulando")
                    self._last_block = cand
                    next_idx += 1
                    continue
                break  # encontrou um passo livre

            if next_idx < len(self.route_ids):
                cand = node_to_coord(self.route_ids[next_idx])
                if cand != self.pos:
                    self.cur_idx = next_idx
                    self.pos = cand
            else:
                # rota ideal esgotou ou só tinha bloqueios – permanece parado
                self.pos = prev_pos

        # detecção de loop ABAB
        self._hist.append(self.pos)
        if len(self._hist) >= 3:
            if (
                self._hist[-1] == self._hist[-3]
                and self._hist[-2] != self._hist[-1]
            ):
                self._ping += 1          # A B A  → ping++
            else:
                self._ping = 0           # padrão quebrou
        # mantém o histórico curto
        if len(self._hist) > 4:
            self._hist.pop(0)

        if self._ping >= PING_LIMIT:
            print(f"[{self.id}] loop ABAB detectado — marcando como incapaz.")
            self.finished  = True
            self.incapable = True
            self.finish_tick = current_tick
            return

        if self.pos == goal:
            self.finished = True
            if self.finish_tick == -1: # Registra o tick de finalização apenas uma vez
                self.finish_tick = current_tick


    def __str__(self) -> str:  # pragma: no cover
        return f"{self.id}: {self.pos}{' ✔' if self.finished else ''}"

# Simulação #
def run_simulation(max_ticks: int = 100) -> None:
    agents = [
        ChatControlledAgent("van‑manhattan", ROUTES["manhattan"], GroqProvider()),
        ChatControlledAgent("van‑euclidean", ROUTES["euclidean"], GeminiProvider()),
        ChatControlledAgent("van‑dijkstra", ROUTES["dijkstra"], GroqProvider2()),
    ]

    print("Tick |     Manhattan     |     Euclidiana     |      Dijkstra")
    print("-----+-------------------+--------------------+--------------")

    last_positions = [ag.pos for ag in agents]
    stuck_counter  = 0

    for tick in range(max_ticks):
        row_out = []
        for ag in agents:
            if not ag.finished:
                ag.step(tick)
            row_out.append(f"{ag.pos[0]}_{ag.pos[1]}")

        print(f"{tick:4d} | {row_out[0]:^17} | {row_out[1]:^19} | {row_out[2]:^15}")

        #  verificação de progresso 
        cur_positions = [ag.pos for ag in agents]
        if cur_positions == last_positions:
            stuck_counter += 1
        else:
            stuck_counter = 0
            last_positions = cur_positions

        if stuck_counter >= STUCK_LIMIT:
            print(f"\nSem progresso há {STUCK_LIMIT} ticks — abortando 🚫\n")
            for ag in agents:
                if not ag.finished:
                    ag.finished   = True
                    ag.incapable  = True
            break

        if all(a.finished for a in agents):
            print("\nTodos os agentes chegaram ao destino! ☕\n")
            break

    for ag in agents:
        if ag.incapable:
            print(f"{ag.id:<15}: Incapaz (loop)   ")
        elif ag.finished:
            print(f"{ag.id:<15}: Chegou no tick {ag.finish_tick}")
        else:
            print(f"{ag.id:<15}: Não chegou em {max_ticks} ticks")


def main() -> None:
    start = time.perf_counter()
    run_simulation(max_ticks=200)
    print(f"Duração total: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
