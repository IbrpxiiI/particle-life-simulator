# src/particle_system.py
"""
ParticleSystem-Klasse für den Particle Life Simulator

Verantwortung von Person A:
- Verwaltung einer Partikelliste
- Integration aller Partikel
- Anwendung von Randbedingungen
- Schnittstellen für Renderer

Aufgaben:
- effiziente Integration vieler Partikel
- performante Randbedingungen
- Vorbereitung für >= 1000 Partikel
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from .particle import Particle
from .interaction_rules import default_rules


class ParticleSystem:
    """
    Verwaltung einer Partikelliste.
    Wichtige Funktionen:
    - integrate(): Geschwindigkeit & Position aktualisieren
    - apply_boundary(): Randverhalten
    """

    def __init__(self, particles: Optional[List[Particle]] = None, rules=None):
        # Wandelt übergebene Partikel in Liste um, falls vorhanden
        self.particles: List[Particle] = list(particles) if particles else []
        self.rules = rules

    # Zufälliges Test System erzeugen
    @classmethod
    def random_system(cls, n=200, num_types=4, width=800, height=600):
        rng = np.random.default_rng()

        # Positionen, v & Typen erzeugen
        positions = rng.uniform([0, 0], [width, height], size=(n, 2))
        velocities = rng.uniform(-1, 1, size=(n, 2))
        types = rng.integers(0, num_types, size=n)

        particles = [
            Particle(tuple(pos), tuple(vel), int(t))
            for pos, vel, t in zip(positions, velocities, types)
        ]

        return cls(particles, default_rules(num_types))

    # Kern: OPTIMIERTE INTEGRATION

    def integrate(self, forces, dt: float = 1.0) -> None:
        """
        Führt den gesamten Integrationsschritt aus.
        Ablauf:
        1. forces -> in numpy Array
        2. Kraft in Beschleunigung umrechnen
        3. Beschleunigung auf v anwenden
        4. Jedes Partikel führt sein eigenes integrate(dt) aus
           -> beinhaltet: Reibung, Noise & Positionsänderung
        """
        if not self.particles:
            return

        forces = np.asarray(forces, dtype=float)
        n = len(self.particles)

        if forces.shape != (n, 2):
            raise ValueError("forces müssen die Form (N,2) haben")

        # Aktuelle v & Massen sammeln
        velocities = np.array([p.velocity for p in self.particles])
        masses = np.array([p.mass if p.mass != 0 else 1.0 for p in self.particles])

        # Beschleunigung berechnen (F = m * a)
        accelerations = forces / masses[:, None]

        # Geschwindigkeit aktualisieren
        velocities += accelerations * dt

        # Ergebnisse zurück in die Partikel schreiben
        for p, v in zip(self.particles, velocities):
            p.velocity = v
            p.integrate(dt)

    # OPTIMIERTE RANDVERARBEITUNG

    def apply_boundary(
        self,
        xlim: Tuple[float, float] = (0.0, 500.0),
        ylim: Tuple[float, float] = (0.0, 500.0),
        mode: str = "clip",
    ) -> None:
        """
        Wendet Randbedingungen an:
        mode:
        "clip" -> Harte Begrenzung, Position wird gekappt
        "wrap" -> Partikel erscheinen auf der anderen Seite
        "reflect" -> Geschwindigkeit wird gespiegelt

        Ablauf:
        Für jeden Partikel wird geprüft:
        - muss die Position verändert werden?
        - muss Velocity geändert werden?
        """

        if not self.particles:
            return

        xmin, xmax = xlim
        ymin, ymax = ylim

        # Positionen & V gesammelt
        positions = np.array([p.position for p in self.particles])
        velocities = np.array([p.velocity for p in self.particles])

        if mode == "clip":
            positions[:, 0] = np.clip(positions[:, 0], xmin, xmax)
            positions[:, 1] = np.clip(positions[:, 1], ymin, ymax)

        elif mode == "wrap":
            width = xmax - xmin
            height = ymax - ymin
            positions[:, 0] = xmin + (positions[:, 0] - xmin) % width
            positions[:, 1] = ymin + (positions[:, 1] - ymin) % height

        elif mode == "reflect":
            # Masken für Kollisionen
            mask_x = (positions[:, 0] < xmin) | (positions[:, 0] > xmax)
            mask_y = (positions[:, 1] < ymin) | (positions[:, 1] > ymax)

            velocities[mask_x, 0] *= -1
            velocities[mask_y, 1] *= -1

            positions[:, 0] = np.clip(positions[:, 0], xmin, xmax)
            positions[:, 1] = np.clip(positions[:, 1], ymin, ymax)

        else:
            raise ValueError("Unbekannter Modus: clip, wrap oder reflect")

        # Ergebnisse zurückschreiben
        for p, pos, vel in zip(self.particles, positions, velocities):
            p.position = pos
            p.velocity = vel

    # # Renderer-Schnittstellen
    def get_positions(self) -> np.ndarray:
        """
        Gibt die Partikelpositionen als Array (N,2) zurück
        """
        return np.array([p.position for p in self.particles])

    def get_types(self) -> np.ndarray:
        """
        Gibt die Partikeltypen aller Partikel zurück
        """
        return np.array([p.type for p in self.particles])
