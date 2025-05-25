from __future__ import annotations
from typing import overload, List, Literal, Union
from pathlib import Path
from statistics import mean
from PIL import Image
from .formatar_imagem_4x4 import formatar_imagem_4x4

TH_OBST = 100        # < cinza → obstáculo
TH_ROAD = 200        # > cinza → via livre

TOKEN_MAP = {
    "OBST": "X",
    "ROAD": "0",
}

@overload
def imagem_para_tokens(
        caminho_in: str | Path,
        tmp_img: str | Path = ...,
        usar_media: Literal[False] = ...
    ) -> List[str]: ...

@overload
def imagem_para_tokens(
        caminho_in: str | Path,
        tmp_img: str | Path = ...,
        usar_media: Literal[True] = ...
    ) -> List[int]: ...

def imagem_para_tokens(
        caminho_in: str | Path,
        tmp_img: str | Path = "tmp_grid.png",
        usar_media: bool = False
    ) -> Union[List[str], List[int]]:
    # print(caminho_in, Path(caminho_in).resolve().exists())

    """
    Gera 16 tokens (linha-a-linha) a partir de uma imagem qualquer.

    Se usar_media=False    → ['0','0','X', ...]  (categorias)
    Se usar_media=True     → [143, 255,  32, ...] (médias)
    """

    # Garante imagem 4×4 gradeada em memória
    formatar_imagem_4x4(caminho_in, tmp_img, dir_celulas=None)
    img = Image.open(tmp_img).convert("L")
    passo = img.size[0] // 4

    tokens = []
    for lin in range(4):
        for col in range(4):
            box = (col*passo, lin*passo, (col+1)*passo, (lin+1)*passo)
            cel = img.crop(box)
            avg = mean(cel.getdata())

            if usar_media:
                tokens.append(int(avg))
            else:
                if avg < TH_OBST:
                    tokens.append(TOKEN_MAP["OBST"])
                elif avg > TH_ROAD:
                    tokens.append(TOKEN_MAP["ROAD"])
                else:                      # zona ambígua → rua sob tráfego*
                    tokens.append("T")     # *o Agente de Controle decide

    return tokens
