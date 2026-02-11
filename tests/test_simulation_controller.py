# tests/test_simulation_controller.py

from __future__ import annotations
from typing import Tuple

import numpy as np
import pygame
import pytest

from src.particle import Particle
from src.particle_system import ParticleSystem
from src.simulation_controller import SimulationController


class DummyRules:
    def __init__(self):
        self.global_strength = 1.0

    def compute_forces(self, system: ParticleSystem):
        n = len(system.particles)
        return np.zeros((n, 2), dtype=float)

    def set_global_strength(self, value: float):
        self.global_strength = float(value)


class DummyRenderer:
    def __init__(self, system: ParticleSystem):
        self.system = system
        self.dt = 0.1
        self.render_calls = 0
        self.last_target_fps = None

    def tick(self, target_fps: int = 60) -> float:
        self.last_target_fps = target_fps
        return self.dt

    def get_fps(self) -> float:
        return 60.0

    def get_fps_avg(self) -> float:
        return 55.0

    def render(self, fps: float | None = None, fps_avg: float | None = None) -> None:
        self.render_calls += 1


def create_simple_system() -> ParticleSystem:
    p = Particle(
        position=(0.0, 0.0),
        velocity=(1.0, 0.0),
        particle_type=0,
        mass=1.0,
        friction=0.0,
        noise=0.0,
    )
    return ParticleSystem(particles=[p])


def test_apply_friction_to_all_sets_all_particles():
    system = create_simple_system()
    rules = DummyRules()
    renderer = DummyRenderer(system)

    controller = SimulationController(
        system=system,
        rules=rules,
        renderer=renderer,
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        boundary_mode="clip",
        target_fps=60,
    )

    controller._apply_friction_to_all(0.123)
    assert controller.global_friction == 0.123
    assert all(abs(p.friction - 0.123) < 1e-9 for p in controller.system.particles)


def test_handle_events_quit_sets_running_false(monkeypatch):
    system = create_simple_system()
    rules = DummyRules()
    renderer = DummyRenderer(system)

    controller = SimulationController(
        system=system,
        rules=rules,
        renderer=renderer,
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        boundary_mode="clip",
        target_fps=60,
    )

    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
    controller.handle_events()
    assert controller.running is False


def test_handle_events_space_toggles_pause(monkeypatch):
    system = create_simple_system()
    rules = DummyRules()
    renderer = DummyRenderer(system)

    controller = SimulationController(
        system=system,
        rules=rules,
        renderer=renderer,
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        boundary_mode="clip",
        target_fps=60,
    )

    controller.paused = False
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)],
    )
    controller.handle_events()
    assert controller.paused is True


def test_handle_events_up_down_changes_global_strength(monkeypatch):
    system = create_simple_system()
    rules = DummyRules()
    renderer = DummyRenderer(system)

    controller = SimulationController(
        system=system,
        rules=rules,
        renderer=renderer,
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        boundary_mode="clip",
        target_fps=60,
    )

    s0 = rules.global_strength

    # UP
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)],
    )
    controller.handle_events()
    assert rules.global_strength > s0

    # DOWN
    s1 = rules.global_strength
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN)],
    )
    controller.handle_events()
    assert rules.global_strength < s1


def test_handle_events_left_right_changes_friction(monkeypatch):
    system = create_simple_system()
    rules = DummyRules()
    renderer = DummyRenderer(system)

    controller = SimulationController(
        system=system,
        rules=rules,
        renderer=renderer,
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        boundary_mode="clip",
        target_fps=60,
    )

    f0 = controller.global_friction

    # RIGHT increases
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)],
    )
    controller.handle_events()
    assert controller.global_friction > f0

    # LEFT decreases (but not below 0)
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)],
    )
    controller.handle_events()
    assert controller.global_friction >= 0.0


def test_step_simulation_moves_particle_when_not_paused():
    system = create_simple_system()
    rules = DummyRules()
    renderer = DummyRenderer(system)

    controller = SimulationController(
        system=system,
        rules=rules,
        renderer=renderer,
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        boundary_mode="clip",
        target_fps=60,
    )

    p_before = system.particles[0].position.copy()
    controller.paused = False
    controller.step_simulation(0.1)
    p_after = system.particles[0].position
    assert p_after[0] > p_before[0]


def test_step_simulation_does_not_move_when_paused():
    system = create_simple_system()
    rules = DummyRules()
    renderer = DummyRenderer(system)

    controller = SimulationController(
        system=system,
        rules=rules,
        renderer=renderer,
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        boundary_mode="clip",
        target_fps=60,
    )

    p_before = system.particles[0].position.copy()
    controller.paused = True
    controller.step_simulation(0.1)
    p_after = system.particles[0].position
    assert p_after[0] == p_before[0]
    assert p_after[1] == p_before[1]


def test_run_loop_calls_renderer_once_and_quits(monkeypatch):
    system = create_simple_system()
    rules = DummyRules()
    renderer = DummyRenderer(system)

    controller = SimulationController(
        system=system,
        rules=rules,
        renderer=renderer,
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        boundary_mode="clip",
        target_fps=120,
    )

    # No events
    monkeypatch.setattr(pygame.event, "get", lambda: [])

    # stop after first render by flipping running to False
    original_render = renderer.render

    def render_and_stop(*args, **kwargs):
        original_render(*args, **kwargs)
        controller.running = False

    renderer.render = render_and_stop

    controller.run()
    assert renderer.render_calls == 1
    assert renderer.last_target_fps == 120