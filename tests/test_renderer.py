# tests/test_renderer.py

import numpy as np

from src.particle import Particle
from src.particle_system import ParticleSystem
from src.renderer import ConsoleRenderer, PygameRenderer


def create_small_system():
    """Helper: create a tiny particle system with 3 particles."""
    particles = [
        Particle(position=(10, 10), velocity=(1, 0), particle_type=0),
        Particle(position=(20, 20), velocity=(0, 1), particle_type=1),
        Particle(position=(30, 30), velocity=(-1, 0), particle_type=2),
    ]
    return ParticleSystem(particles=particles)


def test_console_renderer_runs_without_error(capsys):
    """ConsoleRenderer should run without raising exceptions."""
    system = create_small_system()
    renderer = ConsoleRenderer(system, limit=2)

    # Just call render once and ensure no exception is raised
    renderer.render(step=0)

    # Optionally check that something was printed
    captured = capsys.readouterr()
    assert "Step 0" in captured.out


def test_pygame_renderer_color_map_known_types():
    """PygameRenderer should return a color for known types."""
    system = create_small_system()
    renderer = PygameRenderer(system, width=200, height=200, show_fps=False)

    for t in [0, 1, 2, 3]:
        color = renderer.type_to_color(t)
        assert isinstance(color, tuple)
        assert len(color) == 3


def test_pygame_renderer_color_map_unknown_type():
    """PygameRenderer should fall back to default color for unknown types."""
    system = create_small_system()
    renderer = PygameRenderer(system, width=200, height=200, show_fps=False)

    color = renderer.type_to_color(99)  # unknown type
    assert isinstance(color, tuple)
    assert len(color) == 3
