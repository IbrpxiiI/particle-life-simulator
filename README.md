# Particle Life Simulator

## Project Idea

This project is a Particle Life simulation where thousands of particles move in a 2D space and interact with each other. Each particle has a type (shown as a color), and the interaction between types is defined in a matrix. Depending on the values in this matrix, particles can attract, repel, or ignore each other.

Even though the rules are simple, the movement of many particles at the same time creates complex and interesting patterns. The goal of this project is to simulate this behavior, make it visible, and keep the performance high enough so that the simulation runs smoothly with many particles.

## What This Project Includes

### 1. Simulation
	•	Particle class with position, velocity, and type
	•	Simulation loop that updates movement and applies interaction forces
	•	Interaction matrix for attraction and repulsion
	•	Adjustable parameters (interaction strength, friction, radius, etc.)
	•	Optional real-time visualization or video output

### 2. Code Quality
	•	Clean and readable code
	•	Docstrings in important classes and functions
	•	Unit tests (about 70% coverage)
	•	GitHub Actions pipeline (automatic tests, linting, formatting)

### 3. Performance
	•	Profiling to find performance issues
	•	Optimization using:
	•	NumPy
	•	better algorithms
	•	optional numba or parallelization
	•	Target: at least 1000–2000 particles running smoothly

### 4. Project Management
	•	GitHub repository with Issues and Kanban board
	•	Development through branches and pull requests
	•	Code reviews inside the team
	•	Regular weekly updates during the project

### 5. Documentation & Presentation
	•	README for users and developers
	•	Architecture overview (diagram)
	•	Final presentation of the project
	•	Complete documentation at the end
	

## Optimization

For the final milestone, the focus was on performance analysis and optimization
of the simulation.

- Profiling was performed to identify performance bottlenecks in the simulation loop
- The main bottleneck was the particle interaction calculation, which scales quadratically
- Optimizations were applied (e.g. improved algorithms and use of NumPy operations)
- The simulation runs stably with at least 1000 particles
- All tests pass successfully and the CI pipeline is green

