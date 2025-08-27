"""
Ant Foraging Simulation (Vicsek-inspired with two pheromone fields)

State variables:
    positions[i]      : (x, y) position of ant i in 2D space
    directions[i]     : heading angle (radians) of ant i
    states[i]         : foraging state of ant i (0 = searching, 1 = carrying food)
    pheromone_food    : pheromone field dropped by food-seeking ants
    pheromone_nest    : pheromone field dropped by nest-seeking ants

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import csv
import os

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("outputs", timestamp)
os.makedirs(output_dir, exist_ok=True)
print(f"Artifacts will be saved in: {output_dir}")

# ----------------------
# PARAMETERS
# ----------------------

# Environment
N_ANTS = 50
BOX_SIZE = 100.0
GRID_RES = 50
DX = BOX_SIZE / GRID_RES

# Simulation
DT = 0.2
N_STEPS = 1000
ANT_SPEED = 3
TURN_ANGLE = 0.1

# Pheromone
DIFFUSION = 0.01
DECAY_FOOD = 0
DECAY_NEST = 0.1
BASE_DROP_FOOD = 10000.0
BASE_DROP_NEST = 5000.0

# Landmarks
FOOD_LOCATION = np.array([75.0, 75.0])
FOOD_RADIUS = 7.0
NEST_LOCATION = np.array([25.0, 25.0])
NEST_RADIUS = 7.0
PHEROMONE_SWITCH_PROB = 0.8
PHEROMONE_DROP_INTERVAL = 5  # drop every n steps
PHEROMONE_SEARCH_RADIUS = 7  # The new parameter, in grid cells

MIN_DIST = 5

# Prepare CSV
csv_file = os.path.join(output_dir, "ant_states_log.csv")
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    # header
    writer.writerow(["step", "ant_id", "x", "y", "state", "direction", "vx", "vy", "mode", "deliveries"])

# Inside your animate function, add this at the end
def log_ant_states(step, positions, states, directions, velocities, modes, deliveries):
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        for i in range(N_ANTS):
            writer.writerow([
                step, i,
                positions[i,0], positions[i,1],
                states[i],
                directions[i],
                velocities[i,0], velocities[i,1],
                modes[i],
                deliveries[i] if isinstance(deliveries, np.ndarray) else deliveries
            ])

# ----------------------
# INITIALIZATION
# ----------------------
rng = np.random.default_rng()

positions = NEST_LOCATION + rng.uniform(-10, 10, size=(N_ANTS, 2))  # small random spread
directions = rng.uniform(0, 2*np.pi, N_ANTS)  # initial headings
states = np.zeros(N_ANTS, dtype=int)          # 0 = searching
state_timer = np.zeros(N_ANTS, dtype=int)
modes = np.zeros(N_ANTS, dtype=int)          # 0 = random walk, 1 = pheromone guided
deliveries = np.zeros(N_ANTS, dtype=int)

# Initialize velocities from headings
velocities = ANT_SPEED * np.column_stack((np.cos(directions), np.sin(directions)))

pheromone_food = np.zeros((GRID_RES, GRID_RES))
pheromone_nest = np.zeros((GRID_RES, GRID_RES))
pheromone_total = np.zeros((GRID_RES, GRID_RES))

# ----------------------
# HELPER FUNCTIONS
# ----------------------

def reflect_boundaries(pos, dirs):
    """ Reflect ants at boundaries and flip their heading angle. """
    for i in range(len(pos)):
        if pos[i,0] < 0:
            pos[i,0] = -pos[i,0]
            dirs[i] = np.pi - dirs[i]
        elif pos[i,0] > BOX_SIZE:
            pos[i,0] = 2*BOX_SIZE - pos[i,0]
            dirs[i] = np.pi - dirs[i]

        if pos[i,1] < 0:
            pos[i,1] = -pos[i,1]
            dirs[i] = -dirs[i]
        elif pos[i,1] > BOX_SIZE:
            pos[i,1] = 2*BOX_SIZE - pos[i,1]
            dirs[i] = -dirs[i]
    return pos, dirs

def diffuse_and_decay(field, decay):
    new_field = field.copy()
    new_field[1:-1,1:-1] += DIFFUSION * (
        field[:-2,1:-1] + field[2:,1:-1] +
        field[1:-1,:-2] + field[1:-1,2:] -
        4*field[1:-1,1:-1]
    )
    new_field -= decay
    return new_field

# Initialize strong nest pheromone near the nest
gx, gy = int(NEST_LOCATION[0]/DX), int(NEST_LOCATION[1]/DX)
pheromone_nest[max(0,gx-3):gx+4, max(0,gy-3):gy+4] = BASE_DROP_NEST * 5  # strong initial patch

# Modified move_rw for single scalar
def move_rw(direction):
    noise = rng.uniform(-TURN_ANGLE, TURN_ANGLE)
    return direction + noise

def move_pg(pos, state):
    if state == 0:
        attraction_field = pheromone_food
        repulsion_field = pheromone_nest
    else:
        repulsion_field = pheromone_food
        attraction_field = pheromone_nest
    
    # Use a search radius of 2 for this example.
    search_radius_cells = int(PHEROMONE_SEARCH_RADIUS)

    # Compute local gradient by sampling from a radial window
    gx = int(pos[0] / DX)
    gy = int(pos[1] / DX)
    gx = np.clip(gx, search_radius_cells, GRID_RES - 1 - search_radius_cells)
    gy = np.clip(gy, search_radius_cells, GRID_RES - 1 - search_radius_cells)
    
    # --- Calculate the ATTRACTION gradient ---
    grad_x_attraction = np.sum(attraction_field[gx + 1 : gx + 1 + search_radius_cells, gy - search_radius_cells : gy + search_radius_cells + 1]) \
                      - np.sum(attraction_field[gx - 1 - search_radius_cells : gx, gy - search_radius_cells : gy + search_radius_cells + 1])
    grad_y_attraction = np.sum(attraction_field[gx - search_radius_cells : gx + search_radius_cells + 1, gy + 1 : gy + 1 + search_radius_cells]) \
                      - np.sum(attraction_field[gx - search_radius_cells : gx + search_radius_cells + 1, gy - 1 - search_radius_cells : gy])
    
    # --- Calculate the REPULSION gradient ---
    grad_x_repulsion = np.sum(repulsion_field[gx + 1 : gx + 1 + search_radius_cells, gy - search_radius_cells : gy + search_radius_cells + 1]) \
                     - np.sum(repulsion_field[gx - 1 - search_radius_cells : gx, gy - search_radius_cells : gy + search_radius_cells + 1])
    grad_y_repulsion = np.sum(repulsion_field[gx - search_radius_cells : gx + search_radius_cells + 1, gy + 1 : gy + 1 + search_radius_cells]) \
                     - np.sum(repulsion_field[gx - search_radius_cells : gx + search_radius_cells + 1, gy - 1 - search_radius_cells : gy])
    
    # --- Combine the gradients ---
    # The final gradient is the attraction vector minus the repulsion vector
    final_grad_x = grad_x_attraction - grad_x_repulsion
    final_grad_y = grad_y_attraction - grad_y_repulsion
    
    grad_angle = np.arctan2(final_grad_y, final_grad_x)
    
    return grad_angle


# Follow pheromone
def follow_pheromone(pos, direction, state, modes):
    new_direction = direction.copy()
    for i in range(N_ANTS):
        # Compute local gradient
        gx = int(pos[i,0]/DX)
        gy = int(pos[i,1]/DX)
        gx = np.clip(gx,1,GRID_RES-2)
        gy = np.clip(gy,1,GRID_RES-2)
        
        if state[i] == 0:
            grad_x = pheromone_food[gx+1, gy] - pheromone_food[gx-1, gy]
            grad_y = pheromone_food[gx, gy+1] - pheromone_food[gx, gy-1]
        else:
            grad_x = pheromone_nest[gx+1, gy] - pheromone_nest[gx-1, gy]
            grad_y = pheromone_nest[gx, gy+1] - pheromone_nest[gx, gy-1]
        
        grad_strength = np.hypot(grad_x, grad_y)

        # Probabilistic mode switching logic
        if modes[i] == 0: # Ant is in random walk mode
            # Check if there is a strong enough trail AND a random check passes
            if grad_strength > 1e-1 and rng.random() < PHEROMONE_SWITCH_PROB:
                modes[i] = 1 # Switch to pheromone-guided
        else: # Ant is already in pheromone-guided mode
            # Revert to random walk only if the trail disappears
            if grad_strength < 1e-1:
                modes[i] = 0
        

        # Movement
        if modes[i] == 1:
            new_direction[i] = move_pg(pos[i], state[i])
        else:
            new_direction[i] = move_rw(direction[i])

    return new_direction, modes

def deposit_pheromone(pos, state, step):
    for i in range(N_ANTS):
        if step % PHEROMONE_DROP_INTERVAL != 0:
            continue

        gx = np.clip(int(pos[i,0] / DX), 1, GRID_RES-2)
        gy = np.clip(int(pos[i,1] / DX), 1, GRID_RES-2)

        if state[i] == 1:  # carrying food
            pheromone_food[gx, gy] += BASE_DROP_FOOD
            pheromone_total[gx, gy] += BASE_DROP_FOOD # Add to the total heatmap

        else:               # searching
            pheromone_nest[gx, gy] += BASE_DROP_NEST
            pheromone_total[gx, gy] += BASE_DROP_FOOD # Add to the total heatmap


def update_states(pos, state, state_timer, deliveries):
    for i in range(N_ANTS):
        # If searching and reaches food, start carrying
        if state[i] == 0 and np.linalg.norm(pos[i] - FOOD_LOCATION) < FOOD_RADIUS:
            state[i] = 1
            state_timer[i] = 0

        # If carrying and reaches nest, drop off immediately
        elif state[i] == 1 and np.linalg.norm(pos[i] - NEST_LOCATION) < NEST_RADIUS:
            state[i] = 0
            state_timer[i] = 0
            deliveries += 1  # delivery counted immediately

        else:
            state_timer[i] += 1
    return state, state_timer, deliveries

def handle_collisions(positions, velocities, N_ANTS, MIN_DIST):
    for i in range(N_ANTS):
        for j in range(i + 1, N_ANTS):
            vec = positions[i] - positions[j]
            dist = np.linalg.norm(vec)

            # --- HARD COLLISION (Bouncing) ---
            if dist < MIN_DIST and dist > 1e-6:
                # First, separate the ants to prevent them from getting stuck
                # Push them back exactly to the MIN_DIST boundary
                overlap_vec = (vec / dist) * (MIN_DIST - dist + 1)
                positions[i] += overlap_vec / 2
                positions[j] -= overlap_vec / 2

                # Now, perform a velocity reflection
                n = vec / dist
                
                v1_n = np.dot(velocities[i], n)
                v2_n = np.dot(velocities[j], n)
                
                v1_t = velocities[i] - v1_n * n
                v2_t = velocities[j] - v2_n * n
                
                new_v1_n = v2_n
                new_v2_n = v1_n
                
                velocities[i] = new_v1_n * n + v1_t
                velocities[j] = new_v2_n * n + v2_t

# ----------------------
# SIMULATION LOOP
# ----------------------

fig, ax = plt.subplots(figsize=(6,6) )

# Initialize velocities
velocities = ANT_SPEED * np.column_stack((np.cos(directions), np.sin(directions)))

def animate(step):
    global positions, directions, states, pheromone_food, pheromone_nest
    global state_timer, modes, deliveries, velocities

    # Deposit pheromone
    deposit_pheromone(positions, states, step)

    # Diffuse pheromones
    pheromone_food[:] = diffuse_and_decay(pheromone_food, DECAY_FOOD)
    pheromone_nest[:] = diffuse_and_decay(pheromone_nest, DECAY_NEST)

    states, state_timer, deliveries = update_states(positions, states, state_timer, deliveries)

    # Get new directions and modes from follow_pheromone
    new_directions, modes = follow_pheromone(positions, directions, states, modes)

    # --- FIX STARTS HERE ---
    # Update velocities based on the ant's mode.
    # This is a more direct way to prevent hovering.
    
    for i in range(N_ANTS):
        # Random walk: update direction and velocity
        directions[i] = new_directions[i]
        velocities[i] = ANT_SPEED * np.array([np.cos(directions[i]), np.sin(directions[i])])
    
    handle_collisions(positions, velocities, N_ANTS, MIN_DIST)

    # Update positions
    positions += velocities * DT

    # Reflect at boundaries
    positions, directions = reflect_boundaries(positions, directions)

    # Clear and redraw
    ax.clear()

    ax.set_xlim(0, BOX_SIZE)
    ax.set_ylim(0, BOX_SIZE)
    ax.set_title(f"Step {step} | Deliveries: {deliveries[0]}")

    vmax_food = np.max(pheromone_food)
    vmax_nest = np.max(pheromone_nest)

    ax.imshow(pheromone_food.T, extent=[0, BOX_SIZE, 0, BOX_SIZE],
            origin="lower", cmap="magma_r", alpha=0.8, vmin=0, vmax=vmax_food)
    ax.imshow(pheromone_nest.T, extent=[0, BOX_SIZE, 0, BOX_SIZE],
            origin="lower", cmap="magma", alpha=0.8, vmin=0, vmax=vmax_nest)

    ax.scatter(positions[:,0], positions[:,1], c=states, cmap="coolwarm", edgecolors="k")
    ax.add_patch(plt.Circle(NEST_LOCATION, 5.0, color="brown", alpha=0.3))
    ax.add_patch(plt.Circle(FOOD_LOCATION, 5.0, color="red", alpha=0.3))

    log_ant_states(step, positions, states, directions, velocities, modes, deliveries)

def on_animation_end(anim):
    """Callback function to close the plot when animation ends"""
    plt.close(anim._fig)

# Create animation with the end callback
ani = animation.FuncAnimation(fig, animate, frames=N_STEPS, interval=50, repeat=False)
ani._stop = ani.event_source.stop  # Store the original stop method
ani.event_source.stop = lambda: (ani._stop(), on_animation_end(ani))  # Override to call our function
ani.save(os.path.join( output_dir, "ant_foraging.mp4"), writer="ffmpeg", fps=20)

# 2. Pheromone Convergence Heatmap
fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 8))
heatmap_plot = ax_heatmap.imshow(pheromone_total.T, extent=[0, BOX_SIZE, 0, BOX_SIZE], origin="lower", cmap="viridis")

ax_heatmap.set_title("Pheromone Convergence Heatmap")
ax_heatmap.set_xlabel("X Position")
ax_heatmap.set_ylabel("Y Position")

cbar = fig_heatmap.colorbar(heatmap_plot)
cbar.set_label("Pheromone Accumulation")

fig_heatmap.savefig(os.path.join(output_dir, "pheromone_convergence.png"))
