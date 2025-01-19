import random

# Import the Input Data 
# Example instance: abz5
num_jobs = 10
num_machines = 10
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
def create_population(population_number):
    tasks = []
    for job_id, job in enumerate(job_tasks):
        for i in range(0, len(job), 2):  # Extract (machine, duration) pairs
            tasks.append((job_id, job[i], job[i + 1]))
    return [random.sample(tasks, len(tasks)) for _ in range(population_number)]

POPULATION_SIZE = 10
population = create_population(POPULATION_SIZE)

   
# Calculates the makespan for a given chromosome.
def calculate_makespan(chromosome):
    machine_time = [0] * num_machines
    job_time = [0] * num_jobs
    for job_id, machine, duration in chromosome:
        start_time = max(machine_time[machine], job_time[job_id])
        machine_time[machine] = start_time + duration
        job_time[job_id] = start_time + duration
    return max(machine_time)

fitness_scores = [calculate_makespan(chromo) for chromo in population]


# Selects parents using tournament selection.
def select_parents(population, fitness_scores):
    selected = []
    for _ in range(len(population)):
        i, j = random.sample(range(len(population)), 2)
        selected.append(population[i] if fitness_scores[i] < fitness_scores[j] else population[j])
    return selected

parents = select_parents(population, fitness_scores)
print(f"Step 4: Selected parents from the population.\n")

# Performs one-point crossover.
def crossover(parent1, parent2):
    if random.random() > 0.8:  # 80% crossover rate
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + [gene for gene in parent2 if gene not in parent1[:point]]
    child2 = parent2[:point] + [gene for gene in parent1 if gene not in parent2[:point]]
    return child1, child2

child1, child2 = crossover(parents[0], parents[1])


# Mutates a chromosome by swapping two tasks.
def mutate(chromosome):
    if random.random() < 0.1:  # 10% mutation rate
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

mutated_child = mutate(child1)
print(f"Step 6: Performed mutation.\n")


# Runs the genetic algorithm for job-shop scheduling.
def genetic_algorithm():
    print("Start Genetic Algorithm")
    pass

if __name__ == "__main__":
    print("Main: Job-Shop Scheduling Genetic Algorithm")
    population = create_population(10) 
    genetic_algorithm()
