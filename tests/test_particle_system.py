# tests/test_particle_system.py

import numpy as np
from src.particle import Particle
from src.particle_system import ParticleSystem


def test_integrate_applies_forces_and_moves():
    """
    Testet, ob das Partikelsystem:
    1. Kräfte korrekt auf die Partikel anwendet
    2. Die resultierende Geschwindigkeit aktualisiert
    3. Die Position entsprechend der Geschwindigkeit verändert
    """
    # 2 Partikel mit bekannter Anfangsposition & geschwindigkeit
    p1 = Particle(position=(1.0, 2.0), velocity=(0.5, -0.5), mass=2.0, friction=0.0)
    p2 = Particle(position=(0.0, 0.0), velocity=(0.0, 0.0), mass=2.0, friction=0.0)

    # Partikelsystem mit genau zwei Partikeln
    system = ParticleSystem([p1, p2])

    forces = np.array([[1.0, 0.0], [-1.0, 0.0]])  # Kraft auf p1  # Kraft auf p2

    # System für eine Zeiteinheit (dt = 1.0) integrieren
    system.integrate(forces, dt=1.0)

    # Erwartete neue Geschwindigkeiten:
    # p1: alte v 0.5  + Beschl. 0.5  = 1.0, p2: alte v 0.0  + Beschl. -0.5 = -0.5
    assert np.allclose(p1.velocity[0], 1.0)
    assert np.allclose(p2.velocity[0], -0.5)

    # Erwartete neue Positionen (dt = 1.0):
    # p1: 1.0 + 1.0   = 2.0,  p2: 0.0 + (-0.5) = -0.5
    assert np.allclose(p1.position[0], 2.0)
    assert np.allclose(p2.position[0], -0.5)


def test_apply_boundary_clip():
    """
    Testet, ob das Partikelsystem die Positionen der Partikel
    korrekt auf die erlaubten Grenzen "clippt".
    Clippen bedeutet:
    - Werte unterhalb der Grenze werden auf Min. gesetzt
    - Werte oberhalb werden auf Max. gesetzt
    """
    # Partikel startet außerhalb der erlaubten Grenzen:
    # x = 600  -> zu groß
    # y = -10  -> zu klein
    p = Particle(position=(600.0, -10.0), velocity=(0.0, 0.0))

    system = ParticleSystem([p])

    # Grenzen: 0 bis 500 in X und Y
    system.apply_boundary(
        xlim=(0.0, 500.0), ylim=(0.0, 500.0), mode="clip"  # harte Begrenzung
    )

    assert 0.0 <= p.position[0] <= 500.0
    assert 0.0 <= p.position[1] <= 500.0
