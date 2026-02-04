# src/interaction_rules.py

import numpy as np
import math


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

        # Ergebnisarray für alle Kräfte
        forces = np.zeros((n, 2), dtype=np.float32)

        if n == 0:
            return forces

        # Sicherstellen, dass Typen gültig sind
        if types.min() < 0 or types.max() >= self.num_types():
            raise ValueError("Partikeltypen außerhalb des zulässigen Bereichs.")

        # Lokale Kopien (schnellerer Zugriff in Schleifen)
        matrix = self.matrix
        min_range = self.min_range
        max_range = self.max_range
        global_strength = self.global_strength

        min_range_sq = min_range * min_range
        max_range_sq = max_range * max_range
        inv_min_range = 1.0 / min_range
        inv_range = 1.0 / (max_range - min_range)

        epsilon = 1e-12  # Schutz vor Division durch 0

        for i in range(n):
            xi, yi = positions[i]
            ti = types[i]

            for j in range(i + 1, n):
                dx = positions[j][0] - xi
                dy = positions[j][1] - yi
                dist_sq = dx * dx + dy * dy

                # Keine Kraft bei identischer Position oder zu großer Distanz
                if dist_sq < epsilon or dist_sq > max_range_sq:
                    continue

                dist = math.sqrt(dist_sq)

                # Richtungsvektor (Einheitsvektor)
                inv_dist = 1.0 / dist
                direction_x = dx * inv_dist
                direction_y = dy * inv_dist

                tj = types[j]

                strength_ij = matrix[ti, tj]
                strength_ji = matrix[tj, ti]

                # Zwei Zonen:
                # 1. sehr nahe → starke Abstoßung (verhindert Überschneidungen)
                # 2. normaler Bereich → Matrix-Interaktion
                if dist_sq < min_range_sq:
                    core_factor = (min_range - dist) * inv_min_range

                    # Abstoßung ist symmetrisch: i bekommt -f, j bekommt +f
                    fx = -core_factor * direction_x * global_strength
                    fy = -core_factor * direction_y * global_strength

                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
                else:
                    # Linear abfallende Stärke zwischen min_range und max_range
                    factor = 1.0 - (dist - min_range) * inv_range

                    # i wird von j beeinflusst (strength_ij)
                    fx_i = strength_ij * factor * direction_x * global_strength
                    fy_i = strength_ij * factor * direction_y * global_strength

                    # j wird von i beeinflusst (strength_ji)
                    fx_j = -strength_ji * factor * direction_x * global_strength
                    fy_j = -strength_ji * factor * direction_y * global_strength

                    forces[i][0] += fx_i
                    forces[i][1] += fy_i
                    forces[j][0] += fx_j
                    forces[j][1] += fy_j

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
