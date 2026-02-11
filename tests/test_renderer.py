from particle import Particle
from particle_system import ParticleSystem
from renderer import ConsoleRenderer, PygameRenderer


def create_small_system():
    particles = [
        Particle(position=(10, 10), velocity=(1, 0), particle_type=0),
        Particle(position=(20, 20), velocity=(0, 1), particle_type=1),
        Particle(position=(30, 30), velocity=(-1, 0), particle_type=2),
    ]
    return ParticleSystem(particles=particles)


def test_console_renderer_runs_without_error(capsys):
    system = create_small_system()
    renderer = ConsoleRenderer(system, limit=2)
    renderer.render(step=0)
    captured = capsys.readouterr()
    assert "Step 0" in captured.out


def test_pygame_renderer_color_map_known_types():
    system = create_small_system()
    renderer = PygameRenderer(system, width=200, height=200, show_fps=False)
    for t in [0, 1, 2, 3]:
        c = renderer.type_to_color(t)
        assert isinstance(c, tuple) and len(c) == 3


def test_pygame_renderer_color_map_unknown_type():
    system = create_small_system()
    renderer = PygameRenderer(system, width=200, height=200, show_fps=False)
    c = renderer.type_to_color(99)
    assert isinstance(c, tuple) and len(c) == 3
