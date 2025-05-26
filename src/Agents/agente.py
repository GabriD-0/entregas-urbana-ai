import sys
import os
sys.path.append(os.path.dirname(__file__))  # garante que possamos importar busca.py
import busca

class Agente:
    def __init__(self, nome, heuristica_func):
        """
        Inicializa um agente com um nome e uma função heurística.
        :param nome: Nome ou identificador do agente.
        :param heuristica_func: Função heurística a ser utilizada pelo agente.
        """
        self.nome = nome
        self.heuristica = heuristica_func

    def planejar_caminho(self, mapa, inicio, destino):
        """
        Planeja o caminho do agente do ponto de inicio até o destino usando A*.
        :param mapa: Matriz de booleans indicando células caminháveis.
        :param inicio: Tupla (linha, coluna) inicial.
        :param destino: Tupla (linha, coluna) de destino.
        :return: Lista de células (tuplas) representando o caminho encontrado.
        """
        return busca.a_star(mapa, inicio, destino, self.heuristica)
