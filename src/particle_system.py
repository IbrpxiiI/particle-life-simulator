# src/particle_system.py
"""
ParticleSystem-Klasse für den Particle Life Simulator

Verantwortung von Person A (Paiman):
- Verwaltung einer Partikelliste
- Integration aller Partikel
- Anwendung von Randbedingungen
- Schnittstellen für Renderer
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

    @classmethod
    def random_system(cls, n=200, num_types=4, width=800, height=600):
        """
        Erzeugt ein zufälliges Partikelsystem.

        Parameter:
        n : Anzahl der Partikel
        num_types : Anzahl der Partikeltypen
        width, height : Größe des Simulationsfelds
        """
        rng = np.random.default_rng()
        particles = []

        for _ in range(n):
            x = rng.uniform(0, width)
            y = rng.uniform(0, height)
            vx = rng.uniform(-1, 1)
            vy = rng.uniform(-1, 1)
            t = rng.integers(0, num_types)

            # Position & Velocity als Tupel übergeben, Typ korrekt
            particles.append(
                Particle(position=(x, y), velocity=(vx, vy), particle_type=int(t))
            )

        # Regeln erzeugen (Standard)
        rules = default_rules(num_types)

        return cls(particles, rules)

    def add_particle(self, p: Particle) -> None:
        """
        Fügt 1 einzelnes Partikel zur Systemliste hinzu
        """
        self.particles.append(p)

    def num_particles(self) -> int:
        """
        Gibt Anzahl der Partikel zurück
        """
        return len(self.particles)

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
        if len(self.particles) == 0:
            return

        # Kräfte in ndarray umwandeln
        forces_arr = np.asarray(forces, dtype=float)
        if forces_arr.shape != (len(self.particles), 2):
            raise ValueError(
                "forces must be shape (N,2) where N == number of particles"
            )

        # 1. Physikschritt: Beschleunigung -> Geschwindigkeit
        for i, p in enumerate(self.particles):
            fx, fy = forces_arr[i]
            accel = np.array([fx, fy], dtype=float) / (p.mass if p.mass != 0 else 1.0)
            p.velocity += accel * dt

        # 2. Jeder Partikel integriert sich selbst
        for p in self.particles:
            p.integrate(dt)

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
        xmin, xmax = float(xlim[0]), float(xlim[1])
        ymin, ymax = float(ylim[0]), float(ylim[1])

        for p in self.particles:
            if mode == "clip":
                # Position wird auf erlaubten Grenzen begrenzt
                p.position[0] = np.clip(p.position[0], xmin, xmax)
                p.position[1] = np.clip(p.position[1], ymin, ymax)

            elif mode == "wrap":
                # z.B. l Rand raus -> r Rand rein
                width = xmax - xmin
                height = ymax - ymin
                p.position[0] = xmin + ((p.position[0] - xmin) % width)
                p.position[1] = ymin + ((p.position[1] - ymin) % height)

            elif mode == "reflect":
                # X-Richtung prüfen
                if p.position[0] < xmin:
                    p.position[0] = xmin
                    p.velocity[0] = -p.velocity[0]
                elif p.position[0] > xmax:
                    p.position[0] = xmax
                    p.velocity[0] = -p.velocity[0]
                # Y-Richtung prüfen
                if p.position[1] < ymin:
                    p.position[1] = ymin
                    p.velocity[1] = -p.velocity[1]
                elif p.position[1] > ymax:
                    p.position[1] = ymax
                    p.velocity[1] = -p.velocity[1]

            else:
                raise ValueError(
                    "Unknown boundary mode: choose 'clip', 'wrap', or 'reflect'"
                )

    # Getter für Renderer
    def get_positions(self) -> np.ndarray:
        """
        Gibt die Partikelpositionen als Array (N,2) zurück
        """
        return np.array([p.position for p in self.particles], dtype=float)

    def get_types(self) -> np.ndarray:
        """
        Gibt die Partikeltypen aller Partikel zurück
        """
        return np.array([p.type for p in self.particles], dtype=int)
