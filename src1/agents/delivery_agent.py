from typing import List, Tuple
from PIL import ImageDraw

from .pathfinding import a_star, Cell, Grid

class DeliveryAgent:
    def __init__(self, agent_id: str, grid: Grid,
                 start: Cell, goal: Cell) -> None:
        self.id = agent_id
        self.grid = grid
        self.start = start
        self.goal = goal
        self.path: List[Cell] = a_star(grid, start, goal)

    # ---------- desenhar rota ------------------------------------------
    def draw(self, img, color: Tuple[int,int,int]=(255,0,0)):
        """Desenha linhas ligando centros das células do path."""
        if not self.path: return img
        side = img.size[0]
        draw = ImageDraw.Draw(img)
        points = []
        for cell in self.path:
            x, y = self._center(cell, side)
            points.append((x, y))
        draw.line(points, fill=color, width=5, joint="curve")
        return img

    @staticmethod
    def _center(cell: Cell, side: int, n: int = 4):
        step = side // n
        return (cell[1]*step + step//2, cell[0]*step + step//2)
