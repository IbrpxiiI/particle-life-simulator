# src/renderer.py

from __future__ import annotations
from typing import Dict, Tuple, Optional
from collections import deque

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
    Shows current FPS and rolling average FPS.
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
        fps_avg_window: int = 30,
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

        self.fps_samples = deque(maxlen=int(fps_avg_window))
        self.fps_avg = 0.0

        if color_map is None:
            self.color_map = self._create_default_color_map()
        else:
            self.color_map = color_map

        self._radius = self.particle_radius

        self.particle_surfaces: Dict[int, pygame.Surface] = {}
        for t, color in self.color_map.items():
            surf = pygame.Surface(
                (2 * self._radius + 1, 2 * self._radius + 1),
                pygame.SRCALPHA,
            )
            pygame.draw.circle(surf, color, (self._radius, self._radius), self._radius)
            self.particle_surfaces[int(t)] = surf

        default_color = (200, 200, 200)
        self.default_surface = pygame.Surface(
            (2 * self._radius + 1, 2 * self._radius + 1),
            pygame.SRCALPHA,
        )
        pygame.draw.circle(
            self.default_surface,
            default_color,
            (self._radius, self._radius),
            self._radius,
        )

    def _create_default_color_map(self) -> Dict[int, Color]:
        return {
            0: (255, 80, 80),
            1: (80, 255, 80),
            2: (80, 80, 255),
            3: (255, 255, 80),
        }

    def type_to_color(self, particle_type: int) -> Color:
        return self.color_map.get(int(particle_type), (200, 200, 200))

    def clear(self) -> None:
        self.screen.fill(self.background_color)

    def draw_particles(self) -> None:
        positions = self.system.get_positions()
        types = self.system.get_types()
        positions = np.atleast_2d(positions)

        r = self._radius
        for pos, t in zip(positions, types):
            x = int(pos[0])
            y = int(pos[1])
            surf = self.particle_surfaces.get(int(t), self.default_surface)
            self.screen.blit(surf, (x - r, y - r))

    def draw_overlay(
        self,
        fps: Optional[float] = None,
        fps_avg: Optional[float] = None,
    ) -> None:
        if not self.show_fps:
            return

        if fps is None:
            fps = self.get_fps()

        if fps_avg is None:
            fps_avg = self.get_fps_avg()

        text = f"FPS: {fps:.1f} | avg: {fps_avg:.1f}"
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

    def render(
        self,
        fps: Optional[float] = None,
        fps_avg: Optional[float] = None,
    ) -> None:
        self.clear()
        self.draw_particles()
        self.draw_overlay(fps=fps, fps_avg=fps_avg)
        pygame.display.flip()

    def tick(self, target_fps: int = 60) -> float:
        ms = self.clock.tick(target_fps)

        fps_now = float(self.clock.get_fps())
        self.fps_samples.append(fps_now)
        if self.fps_samples:
            self.fps_avg = sum(self.fps_samples) / len(self.fps_samples)

        return ms / 1000.0

    def get_fps(self) -> float:
        return float(self.clock.get_fps())

    def get_fps_avg(self) -> float:
        return float(self.fps_avg)