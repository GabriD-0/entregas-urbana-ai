"""
Uso:   python -m src.main
Gera 3 arquivos em src/imgs/:
  original.png         (cópia fiel)
  grid_4x4.png         (com grade desenhada)
  routes.png           (grade + rotas dos 2 agentes)
"""
from pathlib import Path
import random

from utils.image_processing import (load_square, overlay_grid,
                                    tokenize, cell_center)
from agents.control_agent import ControlAgent

# ---------- parâmetros ----------------------
IMG_DIR   = Path(__file__).parent / "imgs"
IMG_DIR.mkdir(exist_ok=True)
RAW_IMG   = IMG_DIR / "map.png"          # coloque o PNG aqui
GRID_IMG  = IMG_DIR / "grid_4x4.png"
ROUTE_IMG = IMG_DIR / "routes.png"

def main():
    # 1. carrega e padroniza
    img0 = load_square(RAW_IMG)
    img0.save(IMG_DIR / "original.png")

    # 2. aplica 4×4
    grid_img = overlay_grid(img0, 4)
    grid_img.save(GRID_IMG)

    # 3. tokens
    tokens = tokenize(img0, 4)
    for row in tokens:
        print(row)

    # 4. controla spawns & rotas
    ctrl = ControlAgent(tokens)
    random.seed(42)      # determinístico p/ teste
    ctrl.spawn_agents(2)

    # 5. desenhar rotas
    route_img = grid_img.copy()
    colors = [(255,0,0), (0,128,255)]
    for ag, col in zip(ctrl.agents, colors):
        route_img = ag.draw(route_img, col)
        print(f"{ag.id}: {ag.start} -> {ag.goal} | len={len(ag.path)}")

    route_img.save(ROUTE_IMG)
    print("Imagens geradas em", IMG_DIR.resolve())

    # 6. (opcional) relatar via Gemini
    ctrl.report()

if __name__ == "__main__":
    main()
