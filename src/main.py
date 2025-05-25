from pathlib import Path
from imageFormat.occupancy_grid import build_cost_grid
from agents.control import ControlAgent, ControlProxy
from agents.delivery import PixelDeliveryAgent
from utils.visual_px import draw_path_px

BASE = Path(__file__).resolve().parent
IMG_CLEAN = BASE / "images" / "limpas" / "image1.png"
IMG_OUT   = BASE / "images" / "alteradas" / "image1_trajeto_px.png"

# 1) Constrói grade de custo (reduzindo resolução 4× p/ performance)
costs, img_small = build_cost_grid(IMG_CLEAN, scale=4)

# 2) Cria controlador (para gerar alertas em runtime, se necessário)
# Gera lista de tokens do mapa base ("0" para célula livre, "1" para obstáculo)
H, W = costs.shape
base_map_tokens = ["0" if c > 0 else "1" for c in costs.flatten()]
ctrl_core  = ControlAgent(base_map_tokens, map_width=W)
controller = ControlProxy(ctrl_core)

# 3) Define pontos de partida e destino (em coordenadas da grade reduzida!)
start1 = (10, 20)   # ajuste livremente
goal1  = (110, 200)
start2 = (170, 20)  # ajuste livremente
goal2  = (20, 300)

agent1 = PixelDeliveryAgent(
    agent_id="van1-px",
    start=start1,
    goal=goal1,
    cost_grid=costs,
    controller=controller,
)
agent2 = PixelDeliveryAgent(
    agent_id="van2-px",
    start=start2,
    goal=goal2,
    cost_grid=costs,
    controller=controller,
)

# (Opcional) Registrar agentes para receber alertas de tráfego:
# controller.subscribe(agent1.id, agent1.on_alert)
# controller.subscribe(agent2.id, agent2.on_alert)

# 4) Simula movimentação dos dois agentes
for _ in range(20_000):      # passos suficientes para ambos concluírem
    if agent1.pos != goal1:
        agent1.step()
    if agent2.pos != goal2:
        agent2.step()
    # Gera alertas aleatórios de tráfego (sem replanejamento complexo)
    ctrl_core.tick()
    if agent1.pos == goal1 and agent2.pos == goal2:
        break

# 5) Desenha trajetos dos dois agentes na imagem
draw_path_px(img_small, agent1.history, IMG_OUT,
             grid_shape=costs.shape,
             path2=agent2.history)

print("Histórico do trajeto (van1-px):", agent1.history[:10], "...", agent1.history[-1])
print("Histórico do trajeto (van2-px):", agent2.history[:10], "...", agent2.history[-1])
