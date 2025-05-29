"""metrics_plotter.py
=====================
Script utilitário para gerar gráficos a partir de um arquivo *metrics.json*.

Como funciona
-------------
1. **Entrada**: arquivo JSON no formato `{ algoritmo: { metrica: valor, ... }, ... }`.
2. **Saída**: um PNG por métrica (bar‑chart), salvos no diretório indicado.
3. **Execução**:

```bash
python metrics_plotter.py metrics.json [output_dir]
```

Se `output_dir` não for passado, as imagens são salvas no mesmo diretório do
JSON.

Exemplo mínimo do JSON esperado::

    {
      "Manhattan":  {"total_steps": 27, "total_time": 3.42, "path_length": 34},
      "Euclidean":  {"total_steps": 25, "total_time": 3.15, "path_length": 32},
      "Dijkstra":   {"total_steps": 31, "total_time": 4.01, "path_length": 38}
    }

Copie/importe o módulo para o seu projeto, ou rode isolado após sua simulação.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt

# Classe utilitária

class MetricsPlotter:
    """Carrega métricas de um JSON e cria gráficos de barras."""

    def __init__(self, metrics: Dict[str, Dict[str, float]]):
        self.metrics = metrics

    #  Factories  #
    @classmethod
    def from_json(cls, path: Path) -> "MetricsPlotter":
        """Instancia a partir de um caminho JSON."""
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("metrics.json precisa ser um objeto/ditto na raiz")
        return cls(data)

    # - Plotagem  #
    def plot_all(self, out_dir: Path) -> None:
        """Gera um gráfico por métrica em *out_dir*."""
        out_dir.mkdir(parents=True, exist_ok=True)

        # Descobre o conjunto completo de métricas presentes
        metric_set: set[str] = set()
        for algo_data in self.metrics.values():
            metric_set.update(algo_data.keys())

        for metric in sorted(metric_set):
            self._plot_metric(metric, out_dir)

    def _plot_metric(self, metric: str, out_dir: Path) -> None:
        algs = list(self.metrics.keys())
        vals = [self.metrics[alg][metric] for alg in algs]

        plt.figure()
        plt.bar(algs, vals)
        plt.title(metric.replace("_", " ").title())
        plt.ylabel(metric.replace("_", " ").title())
        plt.tight_layout()

        out_file = out_dir / f"{metric}.png"
        plt.savefig(out_file)
        plt.close()
        print(f"✓ {out_file.relative_to(Path.cwd())}")

# CLI 

def _usage() -> str:
    return (
        "Uso:\n"
        "    python metrics_plotter.py metrics.json [output_dir]\n\n"
        "Argumentos:\n"
        "    metrics.json   Arquivo com as estatísticas.\n"
        "    output_dir     (Opcional) Onde salvar os PNGs. Padrão = pasta do JSON."
    )

def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    if len(argv) < 1:
        sys.exit(_usage())

    metrics_path = Path(argv[0]).expanduser().resolve()
    out_dir = (
        Path(argv[1]).expanduser().resolve() if len(argv) >= 2 else metrics_path.parent
    )

    try:
        plotter = MetricsPlotter.from_json(metrics_path)
        plotter.plot_all(out_dir)
    except Exception as exc:
        sys.exit(f"Erro: {exc}")

if __name__ == "__main__":
    main()
