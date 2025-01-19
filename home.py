import random

# Import the Input Data 
# Example instance: abz5
number_of_jobs = 10
number_of_machines = 10
job_tasks = [
    [4, 88, 8, 68, 6, 94, 5, 99, 1, 67, 2, 89, 9, 77, 7, 99, 0, 86, 3, 92],
    [5, 72, 3, 50, 6, 69, 4, 75, 2, 94, 8, 66, 0, 92, 1, 82, 7, 94, 9, 63],
    [9, 83, 8, 61, 0, 83, 1, 65, 6, 64, 5, 85, 7, 78, 4, 85, 2, 55, 3, 77],
    [7, 94, 2, 68, 1, 61, 4, 99, 3, 54, 6, 75, 5, 66, 0, 76, 9, 63, 8, 67],
    [3, 69, 4, 88, 9, 82, 8, 95, 0, 99, 2, 67, 6, 95, 5, 68, 7, 67, 1, 86],
    [1, 99, 4, 81, 5, 64, 6, 66, 8, 80, 2, 80, 7, 69, 9, 62, 3, 79, 0, 88],
    [7, 50, 1, 86, 4, 97, 3, 96, 0, 95, 8, 97, 2, 66, 5, 99, 6, 52, 9, 71],
    [4, 98, 6, 73, 3, 82, 2, 51, 1, 71, 5, 94, 7, 85, 0, 62, 8, 95, 9, 79],
    [0, 94, 6, 71, 3, 81, 7, 85, 1, 66, 2, 90, 4, 76, 5, 58, 8, 93, 9, 97],
    [3, 50, 0, 59, 1, 82, 8, 67, 7, 56, 9, 96, 6, 58, 4, 81, 5, 59, 2, 96]
]

# Creates a random population of task sequences (chromosomes).
def create_population(pop_size):
    print("Create Initial Population")
    pass

# Calculates the makespan for a given chromosome.
def calculate_makespan(chromosome):
    print("Calculate makespan")
    pass

# Selects parents using tournament selection.
def select_parents(population, fitness_scores):
    print("Perform Selection")
    pass

# Performs one-point crossover.
def crossover(parent1, parent2):
    print("Perform Crossover")
    pass

# Mutates a chromosome by swapping two tasks.
def mutate(chromosome):
    print("Perform Mutation")
    pass

# Runs the genetic algorithm for job-shop scheduling.
def genetic_algorithm():
    print("Start Genetic Algorithm")
    pass

if __name__ == "__main__":
    print("Main: Job-Shop Scheduling Genetic Algorithm")
    parse_input()
    population = create_population(10) 
    genetic_algorithm()
