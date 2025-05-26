import heapq

def heuristica_manhattan(atual, destino):
    """Heurística Manhattan: distância ortogonal (grid) entre duas células."""
    return abs(atual[0] - destino[0]) + abs(atual[1] - destino[1])

def heuristica_euclidiana(atual, destino):
    """Heurística Euclidiana: distância em linha reta entre duas células."""
    return ((atual[0] - destino[0])**2 + (atual[1] - destino[1])**2) ** 0.5


def a_star(mapa_walk, mapa_custo, start, goal, heuristica):
    """
    Algoritmo A* para caminho mínimo em grade.
    ─────────────────────────────────────────────────────────────────────────
    • `mapa_walk`  → matriz bool   (True = caminhável, False = bloqueado)
    • `mapa_custo` → matriz float  (custo de entrar em cada célula)
    • `start`/`goal` → tuplas (linha, coluna)
    • `heuristica`  → função heurística admissível
    Retorna lista de células (do início ao destino) ou [] se não houver rota.
    """
    # Verificações básicas
    if not mapa_walk[start[0]][start[1]] or not mapa_walk[goal[0]][goal[1]]:
        return []          # início ou destino bloqueados

    # Fila de prioridade: (f = g + h, célula)
    fronteira = [(heuristica(start, goal), start)]
    # Costos g(n) (distância real) + predecessores para reconstruir caminho
    g_custo      = {start: 0.0}
    predecessor  = {start: None}
    visitados    = set()

    # Movimentos ortogonais (4-neighbour)
    passos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    H, W   = len(mapa_walk), len(mapa_walk[0])

    while fronteira:
        f_atual, atual = heapq.heappop(fronteira)
        if atual in visitados:
            continue
        visitados.add(atual)

        # Chegamos?
        if atual == goal:
            caminho = []
            n = atual
            while n is not None:
                caminho.append(n)
                n = predecessor[n]
            return caminho[::-1]        # inverte lista

        r, c = atual
        for dr, dc in passos:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if not mapa_walk[nr][nc]:        # bloqueado
                continue

            custo_celula = mapa_custo[nr][nc]
            g_tentativo  = g_custo[atual] + custo_celula

            vizinho = (nr, nc)
            if vizinho not in g_custo or g_tentativo < g_custo[vizinho]:
                g_custo[vizinho] = g_tentativo
                predecessor[vizinho] = atual
                f_vizinho = g_tentativo + heuristica(vizinho, goal)
                heapq.heappush(fronteira, (f_vizinho, vizinho))

    # Nenhum caminho encontrado
    return []
