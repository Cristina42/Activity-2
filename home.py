import random

datasets = {
    "abz5": {
        "num_jobs": 10,
        "num_machines": 10,
        "tasks": [
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
    },


    "la22": {
        "num_jobs": 15,
        "num_machines": 10,
        "tasks": [
            [9, 66, 5, 91, 4, 87, 2, 94, 7, 21, 3, 92, 1, 7, 0, 12, 8, 11, 6, 19],
            [3, 13, 2, 20, 4, 74, 1, 94, 6, 66, 0, 75, 7, 77, 5, 16, 7, 95, 8, 7],
            [8, 77, 7, 20, 2, 34, 0, 15, 9, 88, 5, 89, 6, 53, 3, 6, 1, 45, 4, 76],
            [3, 82, 2, 74, 6, 88, 4, 52, 5, 68, 9, 9, 0, 98, 8, 52, 1, 88, 7, 6],
            [4, 88, 6, 15, 1, 52, 2, 61, 7, 54, 0, 62, 8, 59, 5, 9, 3, 90, 9, 5],
            [4, 71, 6, 41, 3, 33, 7, 91, 5, 78, 9, 2, 8, 23, 0, 29, 2, 72, 1, 50],
            [3, 95, 9, 36, 6, 56, 2, 55, 0, 42, 1, 23, 5, 25, 7, 17, 8, 6],
            [4, 65, 1, 8, 8, 85, 0, 71, 7, 65, 6, 28, 5, 73, 9, 27, 2, 95],
            [3, 68, 7, 35, 8, 9, 5, 55, 0, 59, 2, 6, 1, 24, 4, 51, 6, 90, 9, 51],
            [3, 32, 8, 15, 6, 83, 4, 44, 7, 53, 0, 16, 2, 46, 1, 25, 5, 82],
            [3, 27, 2, 10, 6, 66, 9, 5, 5, 17, 4, 39, 8, 35, 1, 95, 7, 3, 0, 60],
            [3, 54, 5, 20, 2, 21, 8, 73, 9, 14, 4, 59, 6, 34, 0, 55, 1, 34, 7, 55],
            [5, 23, 2, 25, 6, 78, 7, 24, 8, 28, 9, 9, 5, 2, 7, 45, 0, 45],
            [3, 60, 7, 37, 8, 87, 9, 48, 3, 71, 5, 29, 4, 12, 5, 2, 4, 12],
            [4, 90, 8, 17, 2, 49, 3, 83, 1, 40, 6, 23, 7, 65, 9, 27, 0, 48]
        ]
    },
    "la07": {
        "num_jobs": 15,
        "num_machines": 5,
        "tasks": [
            [0, 47, 4, 57, 1, 71, 3, 96, 2, 14],
            [0, 75, 1, 60, 4, 22, 3, 79, 2, 65],
            [3, 32, 0, 33, 2, 69, 1, 31, 4, 58],
            [0, 44, 1, 34, 4, 51, 3, 58, 2, 47],
            [3, 29, 1, 44, 0, 62, 2, 17, 4, 8],
            [1, 15, 2, 40, 0, 97, 4, 38, 3, 66],
            [2, 58, 3, 57, 4, 23, 0, 55, 1, 50],
            [2, 57, 3, 32, 4, 87, 0, 63, 1, 21],
            [4, 56, 0, 84, 3, 58, 2, 61, 1, 60],
            [4, 15, 0, 20, 1, 67, 3, 30, 2, 70],
            [4, 84, 0, 82, 1, 23, 2, 45, 3, 38],
            [3, 50, 2, 21, 4, 23, 0, 44, 1, 29],
            [4, 16, 1, 52, 0, 52, 2, 38, 3, 54],
            [4, 37, 0, 54, 3, 57, 2, 74, 1, 62],
            [4, 57, 1, 61, 0, 81, 2, 30, 3, 68]
    ]
}
}

# Load dataset
selected_dataset = datasets["abz5"]
num_jobs = selected_dataset["num_jobs"]
num_machines = selected_dataset["num_machines"]
job_tasks = selected_dataset["tasks"]

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


# Runs the genetic algorithm for job-shop scheduling.
def genetic_algorithm():
    pop = create_population(POPULATION_SIZE)
    best_chromosome = None
    best_fitness = float('inf')

    for generation in range(100):  # Run for 100 generations
        fitnesses = [calculate_makespan(chromo) for chromo in pop]
        min_fitness = min(fitnesses)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_chromosome = pop[fitnesses.index(min_fitness)]

        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        selected = select_parents(pop, fitnesses)
        next_pop = []
        for i in range(0, len(selected), 2):
            p1, p2 = selected[i], selected[(i + 1) % len(selected)]
            c1, c2 = crossover(p1, p2)
            next_pop.extend([mutate(c1), mutate(c2)])
        pop = next_pop

    return best_chromosome, best_fitness

if __name__ == "__main__":
    for dataset_name, dataset in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Load dataset
        num_jobs = dataset["num_jobs"]
        num_machines = dataset["num_machines"]
        job_tasks = dataset["tasks"]

        # Run the genetic algorithm
        best_schedule, best_makespan = genetic_algorithm()

        # Print results for the current dataset
        print(f"\nDataset: {dataset_name}")
        print("Best Schedule:")
        for task in best_schedule:
            print(f"Job {task[0]} on Machine {task[1]}: Duration {task[2]}")
        print(f"Optimal Makespan: {best_makespan}")

