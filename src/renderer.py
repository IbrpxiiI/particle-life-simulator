# src/renderer.py

from __future__ import annotations
from typing import Dict, Tuple, Optional

import numpy as np
import pygame
from src.particle_system import ParticleSystem

Color = Tuple[int, int, int]


class ConsoleRenderer:
    """
    Very simple renderer that prints particle positions to the console.
    Useful for early debugging without any GUI.
    """

    def __init__(self, system: ParticleSystem, limit: int = 5):
        self.system = system
        self.limit = int(limit)

    def render(self, step: int) -> None:
        positions = self.system.get_positions()
        types = self.system.get_types()
        n = len(positions)

        print(f"Step {step} - showing first {min(self.limit, n)} particles:")
        for i in range(min(self.limit, n)):
            x, y = positions[i]
            t = types[i]
            print(f"  [{i}] type={t}, pos=({x:.2f}, {y:.2f})")
        print("-" * 40)


class PygameRenderer:
    """
    Real-time visualization using pygame.

    Responsibilities:
    - open a window
    - draw all particles as colored circles
    - optionally show FPS counter
    """

    def __init__(
        self,
        system: ParticleSystem,
        width: int = 800,
        height: int = 600,
        background_color: Color = (0, 0, 0),
        particle_radius: int = 3,
        show_fps: bool = True,
        color_map: Optional[Dict[int, Color]] = None,
    ):
        self.system = system
        self.width = int(width)
        self.height = int(height)
        self.background_color = background_color
        self.particle_radius = int(particle_radius)
        self.show_fps = bool(show_fps)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Particle Life")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        if color_map is None:
            self.color_map = self._create_default_color_map()
        else:
            self.color_map = color_map

    # ------------------------------------------------------------------
    def _create_default_color_map(self) -> Dict[int, Color]:
        """
        Default mapping: type -> RGB color.
        """
        return {
            0: (255, 80, 80),  # red-ish
            1: (80, 255, 80),  # green-ish
            2: (80, 80, 255),  # blue-ish
            3: (255, 255, 80),  # yellow-ish
        }

    def type_to_color(self, particle_type: int) -> Color:
        """Returns a color for a given particle type."""
        return self.color_map.get(int(particle_type), (200, 200, 200))

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Fill the screen with the background color."""
        self.screen.fill(self.background_color)

    def draw_particles(self) -> None:
        """
        Draw all particles as circles.
        Uses positions and types from the ParticleSystem.
        """
        positions = self.system.get_positions()
        types = self.system.get_types()

        positions = np.atleast_2d(
            positions
        )  # sicherstellen, dass pos[0]/pos[1] funktioniert

        for pos, t in zip(positions, types):
            x = int(pos[0])
            y = int(pos[1])
            color = self.type_to_color(int(t))
            pygame.draw.circle(self.screen, color, (x, y), self.particle_radius)

    def draw_overlay(self, fps: Optional[float] = None) -> None:
        """Draw optional FPS text."""
        if not self.show_fps or fps is None:
            return

        text_surface = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

    # ------------------------------------------------------------------
    def render(self, fps: Optional[float] = None) -> None:
        """
        Full draw step:
        - clear screen
        - draw particles
        - draw overlay (FPS)
        - flip buffers
        """
        self.clear()
        self.draw_particles()
        self.draw_overlay(fps=fps)
        pygame.display.flip()

    # ------------------------------------------------------------------
    def tick(self, target_fps: int = 60) -> float:
        """
        Limit to target FPS and return dt in seconds.
        """
        ms = self.clock.tick(target_fps)
        dt = ms / 1000.0
        return dt

    def get_fps(self) -> float:
        """Current FPS measured by pygame."""
        return float(self.clock.get_fps())
