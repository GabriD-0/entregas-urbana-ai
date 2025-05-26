import busca

class Agente:
    def __init__(self, nome, heuristica_func):
        self.nome = nome
        self.heuristica = heuristica_func

    def planejar_caminho(self, mapa_walk, mapa_custo, inicio, destino):
        return busca.a_star(mapa_walk, mapa_custo, inicio, destino, self.heuristica)

