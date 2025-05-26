import cv2
import numpy as np
import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "Agents"))
import busca 
from agente import Agente

# Parâmetros da grade e limites para classificação
GRID_ROWS = 4
GRID_COLS = 4
WATER_THRESHOLD = 100    # limiar de pixels de água para considerar célula como rio
BUILDING_THRESHOLD = 5000 # limiar de pixels de prédio para considerar célula como predio
EDGE_THRESHOLD = 1000     # limiar de pixels de aresta para considerar célula como rua

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
    Classifica cada célula da imagem cortada em 'rua', 'rio', 'predio' ou 'outro'.
    Retorna a matriz de classificações e a matriz booleana de caminhabilidade.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # versão em tons de cinza para detecção de bordas
    # Separa os canais de cor
    b, g, r = cv2.split(img)
    # Cria máscara de pixels de água (rios) – cor azul/ciano clara
    mask_water = (((b > 200) & (g > 200) & (r < 150))).astype(np.uint8) * 255
    # Cria máscara de pixels de prédio – cor bege clara (R e G altos, maiores que B)
    mask_building = ((r > b) & (g > b) & (r > 200) & (g > 200)).astype(np.uint8) * 255

    classificacao = [[None]*cols for _ in range(rows)]
    walkable = [[False]*cols for _ in range(rows)]
    # Percorre cada célula da grid
    for i in range(rows):
        for j in range(cols):
            y0 = i * cell_h
            x0 = j * cell_w
            # Conta pixels de rio e prédio na célula usando as máscaras
            water_pixels = cv2.countNonZero(mask_water[y0:y0+cell_h, x0:x0+cell_w])
            building_pixels = cv2.countNonZero(mask_building[y0:y0+cell_h, x0:x0+cell_w])
            # Detecta arestas (bordas) na célula para identificar estruturas lineares (ruas)
            cell_gray = gray[y0:y0+cell_h, x0:x0+cell_w]
            edges = cv2.Canny(cell_gray, 50, 150)
            edge_pixels = cv2.countNonZero(edges)
            # Aplica as regras de classificação com prioridade: rio > prédio > rua > outro
            if water_pixels > WATER_THRESHOLD:
                classificacao[i][j] = 'rio'
                walkable[i][j] = False
            elif building_pixels > BUILDING_THRESHOLD:
                classificacao[i][j] = 'predio'
                walkable[i][j] = False
            elif edge_pixels > EDGE_THRESHOLD:
                classificacao[i][j] = 'rua'
                walkable[i][j] = True
            else:
                classificacao[i][j] = 'outro'
                walkable[i][j] = False
    return classificacao, walkable

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
    caminho_imagem = "src/imgs/map.png"  # Exemplo de nome para imagem fornecida
    imagem = carregar_imagem(caminho_imagem)

    # Prepara a imagem para a grid 4x4
    imagem_cort, cell_h, cell_w = recortar_para_grid(imagem, GRID_ROWS, GRID_COLS)
    cv2.imwrite("src/imgs/original.png", imagem_cort)  # salva imagem original cortada (sem grid)

    # Gera imagem com grid desenhada
    img_grid = imagem_cort.copy()
    desenhar_grid(img_grid, GRID_ROWS, GRID_COLS, cell_h, cell_w)
    cv2.imwrite("src/imgs/grid.png", img_grid)

    # Classifica cada célula da malha
    classificacao, walkable = classificar_celulas(imagem_cort, GRID_ROWS, GRID_COLS, cell_h, cell_w)
    print("Classificação das células:")
    for i in range(GRID_ROWS):
        # Imprime cada linha da matriz de classificação
        print(" ".join(f"{classificacao[i][j]:6}" for j in range(GRID_COLS)))

    # Encontra clusters de ruas conectadas e escolhe posições de início e fim
    clusters = encontrar_clusters_rodoviarios(walkable)
    start1, start2, dest = escolher_pontos(clusters)
    print(f"Start1: {start1}, Start2: {start2}, Destino: {dest}")

    # Inicializa agentes com heurísticas diferentes
    agente1 = Agente("Agente 1", busca.heuristica_manhattan)
    agente2 = Agente("Agente 2", busca.heuristica_euclidiana)

    # Planeja caminhos para cada agente do seu start até o destino comum
    caminho1 = agente1.planejar_caminho(walkable, start1, dest)
    caminho2 = agente2.planejar_caminho(walkable, start2, dest)

    # Log da sequência de células visitadas em cada caminho
    print("Agente 1 caminho:", " -> ".join(str(c) for c in caminho1) if caminho1 else "(nenhum caminho)")
    print("Agente 2 caminho:", " -> ".join(str(c) for c in caminho2) if caminho2 else "(nenhum caminho)")

    # Desenha as rotas na imagem e salva o resultado
    img_routes = imagem_cort.copy()
    desenhar_rotas(img_routes, [caminho1, caminho2], cell_h, cell_w)
    cv2.imwrite("src/imgs/routes.png", img_routes)
