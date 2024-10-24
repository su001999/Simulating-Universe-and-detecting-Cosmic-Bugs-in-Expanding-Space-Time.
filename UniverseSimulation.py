import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Constants for simulation
n_chunks = 10  # Number of initial space-time chunks
num_steps = 50  # Reduced for testing (increase later)
expansion_rate = 1.1  # Rate of universe expansion per step

# Initialize space (S) and time (T) chunks
space_chunks = np.random.choice([0, 1], size=(n_chunks,))
time_chunks = np.random.choice([0, 1], size=(n_chunks,))


# Function for logical disjunction (OR operation)
def disjunction(U_theta_1, U_theta_2):
    return U_theta_1 | U_theta_2


# Function to create the space-time matrix based on SnTn
def create_space_time_matrix(space_chunks, time_chunks):
    matrix = np.zeros((len(space_chunks), len(time_chunks)), dtype=int)
    for i in range(len(space_chunks)):
        for j in range(len(time_chunks)):
            matrix[i, j] = disjunction(space_chunks[i], time_chunks[j])
    return matrix


# Function to evolve space-time matrix with expansion
def expand_space_time_matrix(space_time_matrix):
    # Expand the matrix to simulate the expansion of the universe
    size_increase = int(space_time_matrix.shape[0] * (expansion_rate - 1))
    expanded_matrix = np.pad(space_time_matrix, ((0, size_increase), (0, size_increase)), mode='constant',
                             constant_values=0)
    return expanded_matrix


# Function to visualize the space-time matrix at each time step
def visualize_space_time(matrix, step):
    plt.imshow(matrix, cmap='binary')
    plt.title(f"Space-Time Matrix at Step {step}")
    plt.colorbar(label='Universal Bits (0 or 1)')
    plt.show()


# Function to simulate folding/compression of space-time
def fold_space_time(space_time_matrix):
    # Simulate folding/compression by shifting matrix rows/columns
    matrix_copy = np.roll(space_time_matrix, shift=1, axis=0)  # Shift rows
    matrix_copy = np.roll(matrix_copy, shift=1, axis=1)  # Shift columns
    return matrix_copy


# Non-additive operator (custom summation for space-time chunks)
def non_additive_sum(matrix):
    # Adjust the logic to compute the non-additive sum based on your interpretation
    return np.sum(matrix)  # This could be customized to reflect the actual information content


# Function to calculate particle wave function ΨP based on space-time chunks
def calculate_wave_function(space_chunks, time_chunks):
    combined_chunks = np.array([disjunction(space_chunks[i], time_chunks[i]) for i in range(len(space_chunks))])
    # Here, return a wave function based on the actual context
    wave_function = np.sum(combined_chunks)  # Customize this based on your interpretation
    return wave_function


# Function to print simulation data at each step
def print_simulation_data(step, space_time_matrix, particle_wave_function):
    print(f"Step {step}:")
    print(f"Space-Time Matrix (Shape: {space_time_matrix.shape}):")
    print(space_time_matrix)
    print(f"Particle Wave Function ΨP: {particle_wave_function}")
    print(f"Non-Additive Value: {non_additive_sum(space_time_matrix)}")
    print(f"Integrated Space-Time Value: {np.sum(space_time_matrix)}")
    print("-" * 50)  # Separator for readability


# Function to detect bugs in the simulation
def detect_bugs(step, space_time_matrix, particle_wave_function):
    errors = []

    # Check matrix dimensions
    expected_size = int(n_chunks * (expansion_rate ** step))
    actual_size = space_time_matrix.shape[0]
    if actual_size != expected_size:
        errors.append(f"Step {step}: Matrix size is {actual_size}, expected {expected_size}.")

    # Check for non-additive value
    non_additive_value = non_additive_sum(space_time_matrix)
    # Remove the constraint that non-additive value must be positive
    if non_additive_value < 0:  # Or adjust this condition to fit your logic
        errors.append(f"Step {step}: Non-additive value is {non_additive_value}, which is unexpected.")

    # Check for wave function value
    if particle_wave_function < 0:  # Adjust this check as needed based on your logic
        errors.append(f"Step {step}: Particle wave function ΨP is negative: {particle_wave_function}.")

    # Check for valid values in space-time matrix
    if not np.all(np.isin(space_time_matrix, [0, 1])):  # This may need adjustment
        errors.append(f"Step {step}: Space-Time matrix contains invalid values: {space_time_matrix}")

    # Print any detected errors
    if errors:
        print("Bugs detected:")
        for error in errors:
            print(error)
    else:
        print(f"Step {step}: No bugs detected.")


# Function to simulate and evolve the universe over time
def simulate_universe(space_chunks, time_chunks, num_steps):
    space_time_matrix = create_space_time_matrix(space_chunks, time_chunks)

    # Create a directory to save frames
    if not os.path.exists('frames'):
        os.makedirs('frames')

    for step in range(num_steps):
        print(f"Step {step}: Evolving Space-Time")

        # Calculate particle wave function at this step
        particle_wave_function = calculate_wave_function(space_chunks, time_chunks)

        # Print detailed simulation data every 10 steps
        if step % 10 == 0:
            print_simulation_data(step, space_time_matrix, particle_wave_function)

        # Detect any bugs in the simulation
        detect_bugs(step, space_time_matrix, particle_wave_function)

        # Save the current space-time matrix as an image
        plt.imshow(space_time_matrix, cmap='binary')
        plt.title(f"Space-Time Matrix at Step {step}")
        plt.colorbar(label='Universal Bits (0 or 1)')
        plt.savefig(f'frames/frame_{step:03d}.png')
        plt.close()

        # Apply space-time evolution: folding, expansion, etc.
        space_time_matrix = fold_space_time(space_time_matrix)
        space_time_matrix = expand_space_time_matrix(space_time_matrix)

        # Expand space-time chunks for next step (simulate expansion)
        space_chunks = np.pad(space_chunks, (0, int(len(space_chunks) * (expansion_rate - 1))), mode='constant',
                              constant_values=0)
        time_chunks = np.pad(time_chunks, (0, int(len(time_chunks) * (expansion_rate - 1))), mode='constant',
                             constant_values=0)


# Function to create the animation from saved frames
def create_video_from_frames():
    os.system("ffmpeg -r 10 -i frames/frame_%03d.png -c:v libx264 -pix_fmt yuv420p universe_simulation.mp4")
    print("Video created from frames as universe_simulation.mp4")


# Run the simulation
simulate_universe(space_chunks, time_chunks, num_steps)

# Create video from saved frames
create_video_from_frames()
