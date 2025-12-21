# src/particle.py
"""
Particle-Klasse für den Particle Life Simulator

Verantwortung von Person A (Paiman):
- Datencontainer für ein einzelnes Partikel
- Physikalische Parameter: Masse, Reibung, Rauschen
- Kleine Hilfsfunktionen: Kräfte anwenden, Bewegung integrieren
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


class Particle:
    """
    Repräsentiert ein einzelnes 2D-Partikel.
    Attribute:

    position : np.ndarray
        2D-Position des Partikels (x, y)

    velocity : np.ndarray
        2D-Geschwindigkeit des Partikels (vx, vy)

    type : int
        Partikeltyp (0–3)

    mass : float
        Masse des Partikels. Bestimmt, wie stark Kräfte wirken.

    friction : float
        Reibung, bremst das Partikel in jdm Zeitschritt

    noise : float
        Zufälliges Rauschen, das Bewegung etwas unvorhersehbar macht

    color : Tuple[int, int, int] | None
        RGB Farbe für die Visualisierung
    """

    def __init__(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float] = (0.0, 0.0),
        particle_type: int = 0,
        mass: float = 1.0,
        friction: float = 0.0,
        noise: float = 0.0,
    ):
        # Position in einen NumPy Array umwandeln
        self.position = np.array(position, dtype=float)

        # Geschwindigkeit ebenfalls als Array speichern
        self.velocity = np.array(velocity, dtype=float)

        self.type = int(particle_type)

        # Physikalische Parameter
        self.mass = float(mass)
        self.friction = float(friction)
        self.noise = float(noise)

        # Farbe, vom Renderer gesetzt
        self.color = None

    def apply_force(self, force: Tuple[float, float]) -> None:
        """
        Wendet eine Kraft F auf das Partikel an.
        Formel: F = m * a  ->  a = F / m
        Heißt:
        - Aus der Kraft wird eine Beschleunigung berechnet
        - Die Geschwindigkeit wird sofort um diese Beschleunigung erhöht
        """
        fx, fy = force  # Kraftkomponenten auspacken

        # Beschleunigung berechnen, falls Masse 0 wäre (sollte nicht passieren), Schutzwert 1.0
        accel = np.array([fx, fy], dtype=float) / (self.mass if self.mass != 0 else 1.0)

        # Geschwindigkeit aufgrund der Kraft erhöhen
        self.velocity += accel

    def integrate(self, dt: float = 1.0) -> None:
        """
        Bewegt das Partikel um eine Zeiteinheit dt weiter.

        Schritte:
        1. Reibung anwenden -> Partikel wird langsamer
        2. Rauschen hinzufügen -> leichte zufällige Bewegungen
        3. Position aktualisieren -> position += v * dt
        """

        # 1. Reibung anwenden
        if self.friction != 0.0:
            self.velocity = self.velocity * (1.0 - float(self.friction))

        # 2. Zufälliges Rauschen hinzufügen
        if self.noise > 0.0:
            self.velocity += np.random.randn(2) * float(self.noise)

        # 3. Position aktualisieren, klassische Bewegungsgleichung: x = x + v * dt
        self.position += self.velocity * float(dt)

    def get_state(self) -> dict:
        """
        Gibt den aktuellen Zustand des Partikels zurück.
        """
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "type": self.type,
            "mass": self.mass,
            "friction": self.friction,
            "noise": self.noise,
        }

    def __repr__(self) -> str:
        """
        String-Darstellung des Partikels
        Hilfreich beim Debuggen in der Konsole
        """
        return (
            f"Particle(type={self.type}, pos={self.position.tolist()}, "
            f"vel={self.velocity.tolist()}, mass={self.mass})"
        )
