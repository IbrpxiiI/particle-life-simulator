# src/simulation_controller.py

from __future__ import annotations
from typing import List, Tuple

import numpy as np

# pygame OPTIONAL importieren (CI-safe)
try:
    import pygame
except ImportError:
    pygame = None


from src.particle import Particle
from src.particle_system import ParticleSystem
from src.interaction_rules import InteractionRules, default_rules
from src.renderer import PygameRenderer, ConsoleRenderer

RendererType = Union[PygameRenderer, ConsoleRenderer]


class SimulationController:
    """
    High-level controller for the particle life simulation.

    Responsibilities:
    - create initial particles and rules
    - run the main loop
    - call: rules.compute_forces(system) -> system.integrate(forces, dt)
    - apply boundary conditions
    - forward rendering to the renderer
    - handle basic keyboard input (pause, parameter changes)
    """

    def __init__(
        self,
        system: ParticleSystem,
        rules: InteractionRules,
        renderer: PygameRenderer,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        boundary_mode: str = "wrap",
        target_fps: int = 60,
    ):
        self.system = system
        self.rules = rules
        self.renderer = renderer
        self.xlim = xlim
        self.ylim = ylim
        self.boundary_mode = boundary_mode
        self.target_fps = int(target_fps)

        self.paused = False
        self.running = True

        self.global_friction = 0.02
        self._apply_friction_to_all(self.global_friction)

    # ------------------------------------------------------------------
    def _apply_friction_to_all(self, value: float) -> None:
        """Set the same friction value for all particles in the system."""
        self.global_friction = float(value)
        for p in self.system.particles:
            p.friction = self.global_friction

    # ------------------------------------------------------------------
    def handle_events(self) -> None:
        """Handle pygame events (quit, keyboard)."""
        if pygame is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    # pause / resume
                    self.paused = not self.paused

                # Change global interaction strength
                elif event.key == pygame.K_UP:
                    self.rules.set_global_strength(self.rules.global_strength * 1.1)
                    print(f"global_strength -> {self.rules.global_strength:.3f}")

                elif event.key == pygame.K_DOWN:
                    self.rules.set_global_strength(self.rules.global_strength * 0.9)
                    print(f"global_strength -> {self.rules.global_strength:.3f}")

                # Change friction for all particles
                elif event.key == pygame.K_RIGHT:
                    self._apply_friction_to_all(self.global_friction + 0.005)
                    print(f"friction -> {self.global_friction:.3f}")

                elif event.key == pygame.K_LEFT:
                    new_value = max(0.0, self.global_friction - 0.005)
                    self._apply_friction_to_all(new_value)
                    print(f"friction -> {self.global_friction:.3f}")

    # ------------------------------------------------------------------
    def step_simulation(self, dt: float) -> None:
        """Perform one simulation step if not paused."""
        if self.paused:
            return

        forces = self.rules.compute_forces(self.system)
        self.system.integrate(forces, dt)
        self.system.apply_boundary(self.xlim, self.ylim, mode=self.boundary_mode)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main loop of the simulation."""
        step = 0
        while self.running:
            dt = 0.0
            if hasattr(self.renderer, "tick"):
                dt = self.renderer.tick(self.target_fps)

            self.handle_events()
            self.step_simulation(dt)

            if hasattr(self.renderer, "get_fps"):
                fps = self.renderer.get_fps()
                self.renderer.render(fps=fps)
            else:
                self.renderer.render(step)

            step += 1

        if pygame is not None:
            pygame.quit()


# ----------------------------------------------------------------------
# Helper functions to create a default setup and run the simulation
# ----------------------------------------------------------------------
def create_random_particles(
    count: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    num_types: int = 4,
) -> List[Particle]:
    """
    Create a list of randomly placed particles with random types.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    rng = np.random.default_rng()

    particles: List[Particle] = []
    for _ in range(count):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)

        vx = rng.uniform(-1.0, 1.0)
        vy = rng.uniform(-1.0, 1.0)

        t = rng.integers(0, num_types)

        p = Particle(
            position=(x, y),
            velocity=(vx, vy),
            particle_type=int(t),
            mass=1.0,
            friction=0.02,
            noise=0.0,
        )
        particles.append(p)

    return particles


def main():
    # Simulation domain
    width, height = 800, 600
    xlim = (0.0, float(width))
    ylim = (0.0, float(height))

    num_types = 4
    num_particles = 500

    particles = create_random_particles(num_particles, xlim, ylim, num_types=num_types)

    system = ParticleSystem(particles=particles)
    rules = default_rules(num_types=num_types)

    renderer = PygameRenderer(system, width=width, height=height, particle_radius=3)

    controller = SimulationController(
        system=system,
        rules=rules,
        renderer=renderer,
        xlim=xlim,
        ylim=ylim,
        boundary_mode="wrap",
        target_fps=60,
    )
    controller.run()


if __name__ == "__main__":
    main()
