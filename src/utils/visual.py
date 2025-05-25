from pathlib import Path
from PIL import Image, ImageDraw

def draw_route(base_img: Path | str,
               path: list[tuple[int,int]],
               out_path: Path | str,
               line_color=(255, 0, 0),
               radius=6):
    
    img = Image.open(base_img).convert("RGB")
    size = img.size[0]             # imagem é quadrada (4×4)
    passo = size // 4

    # converte (lin, col)  → (x, y) em pixels
    centers = [(c*passo + passo//2, r*passo + passo//2) for r, c in path]

    draw = ImageDraw.Draw(img)
    draw.line(centers, fill=line_color, width=4, joint="curve")
    for x, y in centers:
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=line_color)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print("Mapa com trajeto salvo em", out_path)
