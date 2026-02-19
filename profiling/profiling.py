import numpy as np

from src.interaction_rules import default_rules
from src.particle import Particle
from src.particle_system import ParticleSystem


def run_profiling(
    n_particles: int = 1500,
    steps: int = 200,
    width: int = 800,
    height: int = 600,
) -> None:
    """
    Run the simulation without rendering for profiling purposes.
    This focuses on force computation + integration + boundary handling.
    """

    rules = default_rules(num_types=4)
    rng = np.random.default_rng(0)

    particles = []
    for i in range(n_particles):
        p_type = i % 4

        position = (float(rng.uniform(0, width)), float(rng.uniform(0, height)))
        velocity = (float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)))

        particles.append(
            Particle(
                position=position,
                velocity=velocity,
                particle_type=p_type,
                mass=1.0,
                friction=0.0,
                noise=0.0,
            )
        )

    ps = ParticleSystem(particles=particles, rules=rules)

    dt = 1.0
    for _ in range(steps):
        forces = rules.compute_forces(ps)
        ps.integrate(forces, dt)
        ps.apply_boundary()


if __name__ == "__main__":
    run_profiling()
