import numpy as np

from src.interaction_rules import default_rules
from src.particle import Particle
from src.particle_system import ParticleSystem


def make_particle(x, y, t):
    return Particle(
        position=(x, y),
        velocity=(0.0, 0.0),
        particle_type=t,
        mass=1.0,
        friction=0.0,
        noise=0.0,
    )


def test_compute_forces_returns_zero_for_single_particle():
    p = make_particle(0.0, 0.0, 0)
    ps = ParticleSystem([p])
    rules = default_rules(num_types=1)

    forces = rules.compute_forces(ps)
    assert forces.shape == (1, 2)
    assert np.allclose(forces, 0.0)


def test_compute_forces_symmetric_particles():
    p1 = make_particle(0.0, 0.0, 0)
    p2 = make_particle(10.0, 0.0, 0)
    ps = ParticleSystem([p1, p2])

    rules = default_rules(num_types=1)
    forces = rules.compute_forces(ps)

    assert np.allclose(forces[0], -forces[1])


def test_set_global_strength_changes_value():
    rules = default_rules(num_types=2)
    rules.set_global_strength(2.5)
    assert rules.global_strength == 2.5


def test_zero_distance_does_not_crash():
    p1 = make_particle(0.0, 0.0, 0)
    p2 = make_particle(0.0, 0.0, 1)
    ps = ParticleSystem([p1, p2])
