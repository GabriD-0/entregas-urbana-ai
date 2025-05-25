from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

def formatar_imagem_4x4(
    caminho_in: str | Path,
    caminho_out: str | Path = "saida_com_grade.png",
    dir_celulas: str | Path | None = "celulas",
    tamanho_final: int = 400,
    cor_grade: tuple[int, int, int] = (128, 128, 128),
    espessura_grade: int = 3,
) -> None:
    """
    Converte a imagem para um tabuleiro 4×4:
      • tons de cinza 8-bit (depois volta a RGB para a grade);
      • recorta centralizado para quadrado preservando proporção;
      • redimensiona para `tamanho_final` px;
      • desenha grade e salva;
      • (opcional) exporta as 16 células.
    """
    caminho_in = Path(caminho_in)
    caminho_out = Path(caminho_out)
    
    # 1. Carregar e cinza
    img = Image.open(caminho_in).convert("L")
    
    # 2. Recortar centralizado + redimensionar de uma vez
    img = ImageOps.fit(
        img,
        size=(tamanho_final, tamanho_final),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    ).convert("RGB")       # volta a RGB para poder colorir a grade
    
    # 3. Desenhar grade
    draw = ImageDraw.Draw(img)
    passo = tamanho_final // 4
    for i in range(1, 4):
        draw.line([(0, i * passo), (tamanho_final, i * passo)],
                  fill=cor_grade, width=espessura_grade)
        draw.line([(i * passo, 0), (i * passo, tamanho_final)],
                  fill=cor_grade, width=espessura_grade)
    
    # 4. Salvar imagem com grade
    caminho_out.parent.mkdir(parents=True, exist_ok=True)
    img.save(caminho_out)
    print("Imagem com grade salva em", caminho_out.resolve())
    
    # 5. (Opcional) exportar células
    if dir_celulas:
        dir_celulas = Path(dir_celulas)
        dir_celulas.mkdir(parents=True, exist_ok=True)
        for lin in range(4):
            for col in range(4):
                box = (col * passo, lin * passo, (col + 1) * passo, (lin + 1) * passo)
                img.crop(box).save(dir_celulas / f"cel_{lin}_{col}.png")
        print("Células salvas em", dir_celulas.resolve())


# --- Exemplo de uso ---
if __name__ == "__main__":
    formatar_imagem_4x4(
        "images/limpas/image1.png",
        "images/alteradas/image1mapa_4x4.png",
        dir_celulas="tiles",
        tamanho_final=400,
        cor_grade=(128, 128, 128),
        espessura_grade=3,
    )
