import cv2
import numpy as np
import random
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "Agents"))

import busca 
from agente import Agente
from segmentation import segmentar

# Parâmetros da grade e limites para classificação
CELL_PX = 32

def salvar_debug(tipo, cell_px, path):
    cores = {
        "rodovia":   (0, 255, 255),     # amarelo → ciano (BGR)
        "rua":       (255, 255, 255),   # branco
        "bloqueado": (50, 50, 50)       # cinza
    }
    rows, cols = tipo.shape
    vis = np.zeros((rows * cell_px, cols * cell_px, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            vis[i*cell_px:(i+1)*cell_px,
                j*cell_px:(j+1)*cell_px] = cores.get(tipo[i, j], (0, 0, 255))
    cv2.imwrite(path, vis)

def carregar_imagem(caminho):
    """Carrega a imagem do mapa a partir do caminho fornecido."""
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")
    return img

def recortar_para_grid(img, rows, cols):
    """
    Corta a imagem de forma que suas dimensões sejam divisíveis por rows x cols.
    Retorna a imagem cortada e as dimensões (altura, largura) de cada célula.
    """
    altura, largura = img.shape[:2]
    nova_altura = (altura // rows) * rows   # maior altura divisível por rows
    nova_largura = (largura // cols) * cols # maior largura divisível por cols
    # Corta a imagem removendo as sobras nas bordas
    img_cortada = img[0:nova_altura, 0:nova_largura].copy()
    cell_h = nova_altura // rows  # altura de cada célula
    cell_w = nova_largura // cols # largura de cada célula
    return img_cortada, cell_h, cell_w

def desenhar_grid(img, rows, cols, cell_h, cell_w):
    """Desenha linhas de grade (grid) na imagem fornecida."""
    altura, largura = img.shape[:2]
    # Linhas horizontais
    for i in range(1, rows):
        y = i * cell_h
        cv2.line(img, (0, y), (largura, y), (0, 0, 0), thickness=1)
    # Linhas verticais
    for j in range(1, cols):
        x = j * cell_w
        cv2.line(img, (x, 0), (x, altura), (0, 0, 0), thickness=1)

def classificar_celulas(img, rows, cols, cell_h, cell_w):
    """
    Baseado nas cores do novo mapa:
    • 'rodovia'  (amarelo)  → caminhável
    • 'rua'      (branca)   → caminhável
    • 'bloqueado'           → tudo o resto
    Retorna (classificacao, walkable, custo)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    classificacao = [[None]*cols for _ in range(rows)]
    walkable      = [[False]*cols for _ in range(rows)]
    custo         = [[999.0]*cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            y0, x0 = i*cell_h, j*cell_w
            tile   = hsv[y0:y0+cell_h, x0:x0+cell_w]
            h_mean, s_mean, v_mean = tile.reshape(-1,3).mean(axis=0)

            # Rodovia amarela
            if 15 <= h_mean <= 40 and s_mean > 80 and v_mean > 130:
                classificacao[i][j] = "rodovia"
                walkable[i][j]      = True
                custo[i][j]         = 0.5

            # Rua branca / marfim
            elif s_mean < 40 and v_mean > 200:
                classificacao[i][j] = "rua"
                walkable[i][j]      = True
                custo[i][j]         = 1.0

            # Bloqueado
            else:
                classificacao[i][j] = "bloqueado"
                # walkable já é False, custo = 999

    return classificacao, walkable, custo

def encontrar_clusters_rodoviarios(walkable):
    """
    Agrupa células caminháveis conectadas ortogonalmente em clusters.
    Retorna lista de clusters, onde cada cluster é uma lista de coordenadas de células.
    """
    clusters = []
    rows, cols = len(walkable), len(walkable[0])
    visitado = [[False]*cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if walkable[i][j] and not visitado[i][j]:
                # Inicia um novo cluster a partir da célula (i,j)
                cluster = []
                stack = [(i, j)]
                visitado[i][j] = True
                while stack:
                    r, c = stack.pop()
                    cluster.append((r, c))
                    # Verifica vizinhos ortogonais
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if walkable[nr][nc] and not visitado[nr][nc]:
                                visitado[nr][nc] = True
                                stack.append((nr, nc))
                clusters.append(cluster)
    return clusters

def escolher_pontos(clusters):
    """
    Escolhe aleatoriamente dois pontos de partida e um destino de um dos clusters fornecidos.
    Garante que o cluster escolhido tenha pelo menos 3 células.
    """
    clusters_validos = [c for c in clusters if len(c) >= 3]
    if not clusters_validos:
        raise Exception("Nenhum cluster >=3 células encontrado.")
    cluster = random.choice(clusters_validos)
    # Seleciona 3 células distintas aleatórias do cluster
    pontos = random.sample(cluster, 3)
    return pontos[0], pontos[1], pontos[2]

def desenhar_rotas(img, paths, cell_h, cell_w):
    """
    Desenha as rotas dos agentes na imagem dada usando cores diferentes.
    `paths` é uma lista de caminhos (listas de células) para cada agente.
    """
    cores = [(0, 0, 255), (0, 255, 0)]  # vermelho, verde em BGR
    for idx, path in enumerate(paths):
        cor = cores[idx % len(cores)]
        # Desenha linhas entre centros das células sequenciais no caminho
        for k in range(len(path) - 1):
            r1, c1 = path[k]
            r2, c2 = path[k+1]
            # Calcula coordenadas do centro de cada célula
            x1 = int(c1 * cell_w + (cell_w - 1) / 2)
            y1 = int(r1 * cell_h + (cell_h - 1) / 2)
            x2 = int(c2 * cell_w + (cell_w - 1) / 2)
            y2 = int(r2 * cell_h + (cell_h - 1) / 2)
            cv2.line(img, (x1, y1), (x2, y2), cor, thickness=3)
        # Desenha círculos nos pontos inicial e final de cada rota
        if path:
            # Início
            x_s = int(path[0][1] * cell_w + (cell_w - 1) / 2)
            y_s = int(path[0][0] * cell_h + (cell_h - 1) / 2)
            cv2.circle(img, (x_s, y_s), radius=5, color=cor, thickness=-1)
            # Destino
            x_e = int(path[-1][1] * cell_w + (cell_w - 1) / 2)
            y_e = int(path[-1][0] * cell_h + (cell_h - 1) / 2)
            cv2.circle(img, (x_e, y_e), radius=5, color=cor, thickness=-1)

# Execução principal do script
if __name__ == "__main__":
    # Caminho da imagem de entrada (pode ser ajustado conforme o nome/locais do arquivo)
    caminho_imagem = "src/imgs/map.png"
    imagem         = carregar_imagem(caminho_imagem)

    # --- nova segmentação por cor -----------------------------------------
    walkable, custo, tipo = segmentar(imagem, cell_px=CELL_PX)

    # salva PNG de depuração (opcional)
    salvar_debug(tipo, CELL_PX, "src/imgs/debug_tiles.png")
    

    rows, cols = walkable.shape
    # “recorta” a imagem para coincidir exatamente com a grade
    imagem_cort = imagem[:rows*CELL_PX, :cols*CELL_PX].copy()

    # Encontra clusters de ruas conectadas e escolhe posições de início e fim
    clusters = encontrar_clusters_rodoviarios(walkable)
    start1, start2, dest = escolher_pontos(clusters)
    print(f"Start1: {start1}, Start2: {start2}, Destino: {dest}")

    # Inicializa agentes com heurísticas diferentes
    agente1 = Agente("Agente 1", busca.heuristica_manhattan)
    agente2 = Agente("Agente 2", busca.heuristica_euclidiana)

    # Planeja caminhos para cada agente do seu start até o destino comum
    caminho1 = agente1.planejar_caminho(walkable, custo, start1, dest)
    caminho2 = agente2.planejar_caminho(walkable, custo, start2, dest)

    # Log da sequência de células visitadas em cada caminho
    print("Agente 1 caminho:", " -> ".join(str(c) for c in caminho1) if caminho1 else "(nenhum caminho)")
    print("Agente 2 caminho:", " -> ".join(str(c) for c in caminho2) if caminho2 else "(nenhum caminho)")

    # Desenha as rotas na imagem e salva o resultado
    img_routes = imagem_cort.copy()
    desenhar_rotas(img_routes, [caminho1, caminho2], CELL_PX, CELL_PX)
    cv2.imwrite("src/imgs/routes.png", img_routes)
