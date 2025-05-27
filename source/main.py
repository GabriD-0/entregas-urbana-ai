from control import ControlAgent
from delivery import DeliveryAgent

grid_rows = grid_cols = 4
ctrl = ControlAgent(rows=grid_rows, cols=grid_cols,
                    ttl_alert=4, max_alerts=3, traffic_penalty=3)

agent = DeliveryAgent("van-01",
                      start_id="14_3", goal_id="2_14",
                      graph_json="source/image_graph.json",
                      control=ctrl)

ctrl.register(agent)

# loop simples de 100 “ticks”
for _ in range(100):
    ctrl.step()
