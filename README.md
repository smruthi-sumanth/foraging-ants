# Ant Foraging Simulation - Parameters and Execution

## How to Run the Simulation

1. **Prerequisites**: Ensure you have Python installed along with the required libraries:
   ```
   pip install numpy matplotlib
   ```
   You'll also need FFmpeg installed on your system to save the animation.

2. **Execution**: Simply run the Python script:
   ```
   python ant_simulation.py
   ```

3. **Output**: The simulation will:
   - Create a timestamped directory in the `outputs` folder
   - Generate an MP4 video of the animation
   - Create a pheromone heatmap visualization
   - Save a detailed CSV log of all ant states
   - Display real-time progress in your console

## Key Simulation Parameters

The simulation offers numerous parameters to customize ant behavior and environment:

### Environment Settings
- `N_ANTS`: Number of ants in the simulation (default: 50)
- `BOX_SIZE`: Size of the simulation area (default: 100.0)
- `GRID_RES`: Resolution of the pheromone grid (default: 50)

### Simulation Control
- `DT`: Time step for each iteration (default: 0.2)
- `N_STEPS`: Total number of simulation steps (default: 1000)
- `ANT_SPEED`: Base movement speed of ants (default: 3)
- `TURN_ANGLE`: Maximum turning angle during random walk (default: 0.1)

### Pheromone System
- `DIFFUSION`: Rate at which pheromones spread (default: 0.01)
- `DECAY_FOOD`: How quickly food pheromones fade (default: 0)
- `DECAY_NEST`: How quickly nest pheromones fade (default: 0.1)
- `BASE_DROP_FOOD`: Amount of food pheromone dropped (default: 10000.0)
- `BASE_DROP_NEST`: Amount of nest pheromone dropped (default: 5000.0)
- `PHEROMONE_SWITCH_PROB`: Chance to follow pheromones when detected (default: 0.8)
- `PHEROMONE_DROP_INTERVAL`: Steps between pheromone deposits (default: 5)
- `PHEROMONE_SEARCH_RADIUS`: Local area ants check for pheromones (default: 7)

### Landmarks
- `FOOD_LOCATION`: Position of the food source (default: [75.0, 75.0])
- `FOOD_RADIUS`: Size of the food source (default: 7.0)
- `NEST_LOCATION`: Position of the nest (default: [25.0, 25.0])
- `NEST_RADIUS`: Size of the nest (default: 7.0)

### Physics
- `MIN_DIST`: Minimum distance before collision occurs (default: 5)
