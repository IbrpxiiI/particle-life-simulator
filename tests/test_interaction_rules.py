import os
import sys

# Sicherstellen, dass der Projektordner im Python-Pfad ist
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
import numpy as np

from src.interaction_rules import InteractionRules, default_rules


class FakeSystem:
    """Ein „Partikelsystem“ nur für Tests.

    Es liefert genau die Methoden, die InteractionRules braucht:
    - get_positions()
    - get_types()
    """

    def __init__(self, positions, types):
        self._positions = np.array(positions, dtype=float)
        self._types = np.array(types, dtype=int)

    def get_positions(self):
        return self._positions

    def get_types(self):
        return self._types


def test_forces_shape_and_type_range():
    """compute_forces gibt ein (N,2)-Array zurück und prüft Typenbereich."""
    # zwei Partikel in 10 Einheiten Abstand
    system = FakeSystem(positions=[(0.0, 0.0), (10.0, 0.0)],
                        types=[0, 1])

    matrix = np.array([[0.5, -1.0],
                       [-1.0, 0.5]])
    rules = InteractionRules(matrix, min_range=5.0, max_range=50.0, global_strength=1.0)

    forces = rules.compute_forces(system)
    assert forces.shape == (2, 2)

    # Typ außerhalb des erlaubten Bereichs -> Fehler
    bad_system = FakeSystem(positions=[(0.0, 0.0), (10.0, 0.0)],
                            types=[0, 3])  # Typ 3 existiert nicht in 2x2-Matrix
    try:
        rules.compute_forces(bad_system)
        assert False, "Es hätte ein ValueError geworfen werden müssen."
    except ValueError:
        pass


def test_cutoff_zero_force_outside_range():
    """Für Distanzen > max_range soll keine Kraft mehr wirken."""
    system = FakeSystem(positions=[(0.0, 0.0), (200.0, 0.0)],
                        types=[0, 1])

    matrix = np.ones((2, 2))
    rules = InteractionRules(matrix, min_range=5.0, max_range=50.0, global_strength=1.0)

    forces = rules.compute_forces(system)
    assert np.allclose(forces, 0.0)


def test_core_repulsion_at_small_distance():
    """Unterhalb min_range soll eine Abstoßung auftreten, auch wenn Matrix 0 ist."""
    # Partikel sehr nah beieinander
    system = FakeSystem(positions=[(0.0, 0.0), (1.0, 0.0)],
                        types=[0, 1])

    # Matrix sagt eigentlich: keine Interaktion
    matrix = np.zeros((2, 2))
    rules = InteractionRules(matrix, min_range=5.0, max_range=50.0, global_strength=1.0)

    forces = rules.compute_forces(system)
    f1, f2 = forces

    # Partikel 2 liegt rechts -> 1 sollte nach links, 2 nach rechts geschoben werden
    assert f1[0] < 0.0
    assert f2[0] > 0.0


def test_default_rules_smoke_test():
    """Smoke-Test: default_rules() funktioniert mit einem einfachen System."""
    system = FakeSystem(
        positions=[(0.0, 0.0), (50.0, 0.0), (30.0, 40.0)],
        types=[0, 1, 2],
    )

    rules = default_rules(num_types=4)
    forces = rules.compute_forces(system)

    assert forces.shape == (3, 2)
    assert np.all(np.isfinite(forces))