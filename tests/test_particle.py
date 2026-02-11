from src.particle import Particle


def test_particle_initialization():
    p = Particle(position=(1.0, 2.0), velocity=(0.5, -0.5), particle_type=1)
    assert p.position[0] == 1.0
    assert p.position[1] == 2.0
    assert p.velocity[0] == 0.5
    assert p.velocity[1] == -0.5
    assert p.type == 1


def test_particle_integrate_moves_particle():
    p = Particle(position=(0.0, 0.0), velocity=(1.0, 0.0), particle_type=0)
    p.integrate(dt=1.0)
    assert p.position[0] > 0.0
