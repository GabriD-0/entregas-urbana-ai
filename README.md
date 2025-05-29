### 📍 Path-Finding Map Pipeline

Pipeline completo para:

1. **converter um mapa estilizado do Google Maps em grade 16 × 16**  
2. **gerar grafos e rotas** com algoritmos heurísticos (A*) e não-heurísticos (Dijkstra)  
3. **executar uma simulação de entrega** com agentes reagindo a tráfego dinâmico  
4. **comparar métricas** em gráficos prontos  
5. _(bonus)_ **controlar agentes com LLMs** via APIs GROQ / Gemini  



## 1. Pré-requisitos

| Item | Versão recomendada |
|------|-------------------|
| Python | **3.10 – 3.12** (testado em 3.12.0) |
| Pip / venv | Última estável |
| GCC / build-essentials | Para compilar dependências nativas do OpenCV |

Crie e ative um ambiente virtual, depois instale todas as dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt
````

---

## 2. Preparar o mapa

1. Acesse **[https://mapstyle.withgoogle.com](https://mapstyle.withgoogle.com)**.
2. Selecione o **tema “Retro”**.
3. No painel **Landmarks → ocultar** e **Labels → None**.
4. Faça zoom/scroll no trecho que deseja (ex.: 16 × 16 quarteirões).
5. Exporte a imagem (PNG) em alta resolução.
6. Renomeie para **`image.png`** e **substitua** o arquivo existente na raiz do projeto (ou `source/image.png` se preferir).

> ⚠️ O repositório já traz um modelo pronto; basta sobrescrevê-lo para usar seu próprio mapa.

---

## 3. Executar o pipeline principal

```bash
python main.py
```

O script executa as etapas:

1. Remove fundo do mapa e extrai apenas as vias.
2. Aplica grade 16 × 16 e rotula cada célula.
3. Constrói o grafo `source/json/image_graph.json`.
4. Calcula três rotas:

   * **A\* Manhattan**
   * **A\* Euclidean**
   * **Dijkstra**
5. Simula **`ticks`** de tráfego dinâmico, gerando históricos em
   `source/json/ticks_routes.json`.
6. Salva todas as imagens em `source/imgs/*`.

---

## 4. Gerar gráficos de métricas

```bash
python metrics_graphs.py
```

Arquivos PNG comparando tempo de planejamento, número de replans, etc. serão criados em:

```
source/imgs/metrics/
```

---

## 5. (Bonus) Modo **LLM Agents**

Crie um arquivo **`.env`** na raiz com suas chaves:

```
GROQ_API_KEY=...
GROQ_API_KEY2=...
GEMINI_API_KEY=...
```

Depois execute:

```bash
python chat.py
```

O script carrega as rotas de `ticks_routes.json` e pede a três LLMs que tomem decisões de
navegação em tempo real — ótimo para experimentar comportamento de IA generativa em
ambientes de trajetórias!

---

## 6. Estrutura de pastas (gerada após a execução)

```
source/
├── image.png                # mapa original (ou o seu)
├── imgs/
│   ├── 1_image_linhas.png
│   ├── 2_image_grid.png
│   ├── 3_image_grid_labels.png
│   └── rotas/
│       ├── 1_image_route_manhattan.png
│       ├── 2_image_route_euclid.png
│       ├── 3_image_route_dijk.png
│       ├── 4_rota_real_manhattan.png
│       ├── 5_rota_real_euclidiana.png
│       └── 6_rota_real_dijkstra.png
├── json/
│   ├── image_graph.json
│   ├── metrics.json
│   └── ticks_routes.json
└── imgs/metrics/            # gráficos do metrics_graphs.py
```

---

## 7. Scripts principais

| Script              | Descrição                                                      |
| ------------------- | -------------------------------------------------------------- |
| `main.py`           | Pipeline completo: processamento de imagem → rotas → simulação |
| `metrics_graphs.py` | Gera gráficos comparativos das rotas                           |
| `chat.py`           | (Opcional) agentes controlados por LLMs usando GROQ / Gemini   |
| `control.py`        | Gerencia alertas de tráfego                                    |
| `delivery.py`       | Lógica de agentes de entrega                                   |
| `pathfinder.py`     | Implementações de A\* e Dijkstra                               |
| `rota_mapa.py`      | Funções de processamento de imagem                             |

---

## 8. Licença

Distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para detalhes.

Boa exploração! 🚚💨

