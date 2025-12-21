# tests/test_simulation_controller.py

from __future__ import annotations
from typing import Tuple, List

import numpy as np

from src.particle import Particle
from src.particle_system import ParticleSystem
from src.simulation_controller import SimulationController


class DummyRules:
    """Simple rules object that always returns zero forces."""

    def __init__(self):
        self.global_strength = 1.0

    def compute_forces(self, system: ParticleSystem):
        n = len(system.particles)
        return np.zeros((n, 2), dtype=float)

    # For compatibility with SimulationController keyboard controls:
    def set_global_strength(self, value: float):
        self.global_strength = float(value)


class DummyRenderer:
    """
    Minimal renderer used for tests.

    Does not open a window and only records calls.
    """

    def __init__(self, system: ParticleSystem):
        self.system = system
        self.dt = 0.1
        self.render_calls = 0

    def tick(self, target_fps: int = 60) -> float:
        return self.dt

    def get_fps(self) -> float:
        return 60.0

    def render(self, fps: float | None = None) -> None:
        self.render_calls += 1


def create_simple_system() -> ParticleSystem:
    """One particle with non-zero velocity."""
    p = Particle(
        position=(0.0, 0.0),
        velocity=(1.0, 0.0),
        particle_type=0,
        mass=1.0,
        friction=0.0,
        noise=0.0,
    )
    return ParticleSystem(particles=[p])


def test_step_simulation_moves_particle_when_not_paused():
    system = create_simple_system()
    rules = DummyRules()
    renderer = DummyRenderer(system)

    xlim: Tuple[float, float] = (0.0, 100.0)
    ylim: Tuple[float, float] = (0.0, 100.0)

    controller = SimulationController(
        system=system,
        rules=rules,
        renderer=renderer,
        xlim=xlim,
        ylim=ylim,
        boundary_mode="clip",
        target_fps=60,
    )

    p_before = system.particles[0].position.copy()
    controller.paused = False

    dt = 0.1
    controller.step_simulation(dt)

    p_after = system.particles[0].position
    # x coordinate should have increased because velocity is (1, 0)
    assert p_after[0] > p_before[0]


def test_step_simulation_does_not_move_when_paused():
    system = create_simple_system()
    rules = DummyRules()
    renderer = DummyRenderer(system)

    xlim: Tuple[float, float] = (0.0, 100.0)
    ylim: Tuple[float, float] = (0.0, 100.0)

    controller = SimulationController(
        system=system,
        rules=rules,
        renderer=renderer,
        xlim=xlim,
        ylim=ylim,
        boundary_mode="clip",
        target_fps=60,
    )

    p_before = system.particles[0].position.copy()
    controller.paused = True

    dt = 0.1
    controller.step_simulation(dt)

    p_after = system.particles[0].position
    # position should be unchanged while paused
    assert p_after[0] == p_before[0]
    assert p_after[1] == p_before[1]
