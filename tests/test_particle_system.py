import numpy as np
import pytest

from src.particle import Particle
from src.particle_system import ParticleSystem


def make_particle(x=0.0, y=0.0, vx=0.0, vy=0.0, t=0, mass=1.0):
    return Particle(
        position=(x, y),
        velocity=(vx, vy),
        particle_type=t,
        mass=mass,
        friction=0.0,
        noise=0.0,
    )


def test_integrate_empty_system_does_not_crash():
    ps = ParticleSystem([])
    ps.integrate(forces=np.zeros((0, 2)), dt=1.0)  # should not raise


def test_integrate_wrong_force_shape_raises():
    ps = ParticleSystem([make_particle(), make_particle()])
    with pytest.raises(ValueError):
        ps.integrate(forces=np.zeros((1, 2)), dt=1.0)


def test_integrate_moves_particle():
    p = make_particle(vx=1.0, vy=0.0)
    ps = ParticleSystem([p])
    ps.integrate(forces=np.array([[0.0, 0.0]]), dt=1.0)
    assert p.position[0] > 0.0


def test_integrate_zero_mass_does_not_crash():
    p = make_particle(mass=0.0, vx=1.0, vy=0.0)
    ps = ParticleSystem([p])
    ps.integrate(forces=[[1.0, 0.0]], dt=1.0)
    assert p.position[0] != 0.0


def test_apply_boundary_clip():
    p = make_particle(x=200.0, y=-10.0)
    ps = ParticleSystem([p])
    ps.apply_boundary(xlim=(0, 100), ylim=(0, 100), mode="clip")
    assert 0.0 <= p.position[0] <= 100.0
    assert 0.0 <= p.position[1] <= 100.0


def test_apply_boundary_wrap():
    p = make_particle(x=150.0, y=150.0)
    ps = ParticleSystem([p])
    ps.apply_boundary(xlim=(0, 100), ylim=(0, 100), mode="wrap")
    assert 0.0 <= p.position[0] < 100.0
    assert 0.0 <= p.position[1] < 100.0


def test_apply_boundary_reflect_flips_velocity_x():
    p = make_particle(x=-1.0, y=50.0, vx=2.0, vy=0.0)
    ps = ParticleSystem([p])
    ps.apply_boundary(xlim=(0, 100), ylim=(0, 100), mode="reflect")
    assert p.position[0] == 0.0
    assert p.velocity[0] == -2.0


def test_apply_boundary_reflect_y():
    p = make_particle(x=50.0, y=-5.0, vx=0.0, vy=3.0)
    ps = ParticleSystem([p])
    ps.apply_boundary(xlim=(0, 100), ylim=(0, 100), mode="reflect")
    assert p.position[1] == 0.0
    assert p.velocity[1] == -3.0


def test_apply_boundary_unknown_mode_raises():
    ps = ParticleSystem([make_particle()])
    with pytest.raises(ValueError):
        ps.apply_boundary(mode="unknown")


def test_get_positions_and_types():
    p1 = make_particle(t=1)
    p2 = make_particle(t=2)
    ps = ParticleSystem([p1, p2])
    assert ps.get_positions().shape == (2, 2)
    assert ps.get_types().tolist() == [1, 2]


def test_random_system_factory_smoke():
    ps = ParticleSystem.random_system(n=10, num_types=3, width=100, height=50)
    assert ps.num_particles() == 10
    assert ps.rules is not None
    assert ps.get_positions().shape == (10, 2)
    assert ps.get_types().shape == (10,)