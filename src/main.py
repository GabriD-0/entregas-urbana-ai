# src/main.py
from pathlib import Path
from imageFormat.image_to_tokens import imagem_para_tokens
from agents.delivery import DeliveryAgent
from agents.control import ControlAgent, ControlProxy

# ────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent           # …/Faculdade AI
IMG_CLEAN = BASE_DIR / "src" / "images" / "limpas" / "image1.png" # foto original
TMP_GRID  = BASE_DIR / "src" / "images" / "alteradas" / "image1mapa_4x4.png"

# Passa a imagem limpa → a função criará TMP_GRID automaticamente
tokens = imagem_para_tokens(IMG_CLEAN, tmp_img=TMP_GRID)

ctrl_core  = ControlAgent(tokens)
controller = ControlProxy(ctrl_core)

delivery = DeliveryAgent(
    agent_id="van-42",
    start=(0, 0),
    goal=(3, 3),
    base_map=tokens,
    controller=controller,
)

for _ in range(50):
    delivery.step()
    ctrl_core.tick()
    if delivery.pos == delivery.goal:
        break
