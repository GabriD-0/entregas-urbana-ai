import json
import matplotlib.pyplot as plt
from pathlib import Path


SRC          = Path("source/json/metrics.json")   # ajuste se necessário
OUT_DIR      = Path("source/imgs/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data         = json.loads(SRC.read_text(encoding="utf-8"))
algs         = ["manhattan", "euclidean", "dijkstra"]
metrics      = [
    "initial_plan_time_s",
    "total_plan_time_s",
    "replan_count",
    "planned_path_len",
    "actual_steps"
]

for m in metrics:
    vals = [data[a][m] for a in algs]
    plt.figure()
    plt.bar(algs, vals)
    plt.ylabel(m)
    plt.title(m.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{m}.png", dpi=120)
    plt.close()
print(f"Gráficos salvos em {OUT_DIR}")
