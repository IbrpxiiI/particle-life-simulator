# src/interaction_rules.py

import numpy as np


class InteractionRules:
    """
    Einfaches, gut verständliches Modul für Partikelinteraktion.
    - matrix[i][j] gibt an, wie stark Typ j auf Typ i wirkt
    - min_range: starke Nah-Abstoßung (verhindert Überschneidungen)
    - max_range: maximale Distanz, ab der keine Kräfte mehr wirken
    - global_strength: Verstärkung aller Kräfte (für GUI/Parameter)
    """

    def __init__(self, matrix, min_range=5.0, max_range=120.0, global_strength=1.0):
        self.matrix = np.array(matrix, dtype=float)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.global_strength = float(global_strength)

        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Interaktionsmatrix muss quadratisch sein (T x T).")

    def num_types(self):
        """Anzahl der Partikeltypen (Dimension der Matrix)."""
        return self.matrix.shape[0]

    def set_global_strength(self, value):
        """Ändert die globale Interaktionsstärke (für GUI)."""
        self.global_strength = float(value)

    def set_ranges(self, min_range, max_range):
        """Erlaubt es, die minimale und maximale Reichweite zur Laufzeit zu ändern."""
        if min_range < 0 or max_range <= min_range:
            raise ValueError("0 <= min_range < max_range muss gelten.")
        self.min_range = float(min_range)
        self.max_range = float(max_range)

    def compute_forces(self, system):
        """
        Berechnet für jedes Partikel die Kraft basierend auf:
        - Typen
        - Abständen
        - Interaktionsmatrix
        - Reichweitenparametern

        Rückgabe: ndarray der Form (N, 2)
        """
        positions = system.get_positions()
        types = system.get_types()
        n = len(positions)

        forces = np.zeros((n, 2), dtype=float)

        if n == 0:
            return forces

        # Sicherstellen, dass Typen gültig sind
        if types.min() < 0 or types.max() >= self.num_types():
            raise ValueError("Partikeltypen außerhalb des zulässigen Bereichs.")

        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[j] - positions[i]
                dist = np.linalg.norm(dx)

                # Keine Kraft wenn zu weit weg oder exakt gleiche Position
                if dist == 0.0 or dist > self.max_range:
                    continue

                direction = dx / dist
                ti = types[i]
                tj = types[j]

                strength_ij = self.matrix[ti, tj]
                strength_ji = self.matrix[tj, ti]

                # zwei Zonen:
                # 1. sehr nahe → starke Abstoßung
                # 2. normaler Bereich → Matrix-Interaktion
                if dist < self.min_range:
                    core_factor = (self.min_range - dist) / self.min_range
                    f_i = -core_factor * direction * self.global_strength
                    f_j = core_factor * direction * self.global_strength
                else:
                    # linear abfallende Stärke
                    factor = 1.0 - (dist - self.min_range) / (
                        self.max_range - self.min_range
                    )
                    f_i = strength_ij * factor * direction * self.global_strength
                    f_j = -strength_ji * factor * direction * self.global_strength

                forces[i] += f_i
                forces[j] += f_j

        return forces


def default_rules(num_types=4):
    """
    Erstellt eine Default-Matrix für 4 Partikeltypen.
    Sie ist nicht trivial, erzeugt aber gut sichtbare Strukturen.
    """

    if num_types == 4:
        matrix = [
            [0.6, -0.8, 0.3, -0.2],
            [-0.5, 0.6, -0.7, 0.1],
            [0.2, -0.4, 0.6, -0.6],
            [-0.3, 0.1, -0.5, 0.6],
        ]
    else:
        rng = np.random.default_rng(0)
        matrix = rng.uniform(-1.0, 1.0, (num_types, num_types))
        np.fill_diagonal(matrix, 0.5)

    return InteractionRules(matrix)
