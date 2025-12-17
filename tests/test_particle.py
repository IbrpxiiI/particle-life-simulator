# tests/test_particle.py
import numpy as np                      
from src.particle import Particle       

def test_particle_initialization():
    """
    Testet, ob ein Partikel nach der Erstellung alle Attribute korrekt speichert
    """
    p = Particle(
        position=(1.0, 2.0),            # Startposition X=1, Y=2
        velocity=(0.5, -0.5),           # Anfangsgeschwindigkeit (r, l)
        particle_type=2,                
        mass=2.0,                       
        friction=0.1,                   # Reibungswert = 0.1
        noise=0.0                       # Kein Rauschen
    )

    assert np.allclose(p.position, np.array([1.0, 2.0]))
    assert np.allclose(p.velocity, np.array([0.5, -0.5]))

    assert p.type == 2
    assert p.mass == 2.0


def test_apply_force_and_integrate():
    """
    Testet:
    1. ob eine Kraft F richtig in Beschleunigung a = F/m umgerechnet wird
    2. ob danach die Geschwindigkeit korrekt steigt
    3. ob nach integrate(dt) die Position richtig aktualisiert wird
    """

    p = Particle(position=(0, 0), velocity=(0, 0), mass=2.0)

    # Kraft F = (2, 0) anwenden
    # Physikalisch: a = F/m = 2/2 = 1 , -> v erhöht sich um +1
    p.apply_force((2.0, 0.0))

    # Einen Zeitschritt dt = 1.0 simulieren: position += v * dt
    p.integrate(dt=1.0)

    # Nach Kraftanwendung sollte v = (1, 0) sein
    assert np.allclose(p.velocity, np.array([1.0, 0.0]))

    # Nach Integrationsschritt sollte position = (1, 0) sein
    assert np.allclose(p.position, np.array([1.0, 0.0]))
