import random
import matplotlib.pyplot as plt

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


    "la36": {
        "num_jobs": 15,
        "num_machines": 15,
        "tasks": [
            [4, 21, 3, 55, 6, 71, 14, 98, 10, 12, 2, 34, 9, 16, 1, 21, 0, 53, 7, 26, 8, 52, 5, 95, 12, 31, 11, 42, 13, 39],
            [11, 54, 4, 83, 1, 77, 7, 64, 8, 34, 14, 79, 12, 43, 0, 55, 3, 77, 6, 19, 9, 37, 5, 79, 10, 92, 13, 62, 2, 66],
            [9, 83, 5, 77, 2, 87, 7, 38, 4, 60, 12, 98, 0, 93, 13, 17, 6, 41, 10, 44, 3, 69, 11, 49, 8, 24, 1, 87, 14, 25],
            [5, 77, 0, 96, 9, 28, 6, 7, 4, 95, 13, 35, 7, 35, 8, 76, 11, 9, 12, 95, 2, 43, 1, 75, 10, 61, 14, 10, 3, 79],
            [10, 87, 4, 28, 8, 50, 2, 59, 0, 46, 11, 45, 14, 9, 9, 43, 6, 52, 7, 27, 1, 91, 13, 41, 3, 16, 5, 59, 12, 39],
            [0, 20, 2, 71, 4, 78, 13, 66, 3, 14, 12, 8, 14, 42, 6, 28, 1, 54, 9, 33, 11, 89, 8, 26, 7, 37, 10, 33, 5, 43],
            [8, 69, 4, 96, 12, 17, 0, 69, 7, 45, 11, 31, 6, 78, 10, 20, 3, 27, 13, 87, 1, 74, 5, 84, 14, 76, 2, 94, 9, 81],
            [4, 58, 13, 90, 11, 76, 3, 81, 7, 23, 9, 28, 1, 18, 2, 32, 12, 86, 8, 99, 14, 97, 0, 24, 10, 45, 6, 72, 5, 25],
            [5, 27, 1, 46, 6, 67, 8, 27, 13, 19, 10, 80, 2, 17, 3, 48, 7, 62, 11, 12, 14, 28, 4, 98, 0, 42, 9, 48, 12, 50],
            [11, 37, 5, 80, 4, 75, 8, 55, 7, 50, 0, 94, 9, 14, 6, 41, 14, 72, 3, 50, 10, 61, 13, 79, 2, 98, 12, 18, 1, 63],
            [7, 65, 3, 96, 0, 47, 4, 75, 12, 69, 14, 58, 10, 33, 1, 71, 9, 22, 13, 32, 5, 57, 8, 79, 2, 14, 11, 31, 6, 60],
            [1, 34, 2, 47, 3, 58, 5, 51, 4, 62, 6, 44, 9, 8, 7, 17, 10, 97, 8, 29, 11, 15, 13, 66, 12, 40, 0, 44, 14, 38],
            [3, 50, 7, 57, 13, 61, 5, 20, 11, 85, 12, 90, 2, 58, 4, 63, 10, 84, 1, 39, 9, 87, 6, 21, 14, 56, 8, 32, 0, 57],
            [9, 84, 7, 45, 5, 15, 14, 41, 10, 18, 4, 82, 11, 29, 2, 70, 1, 67, 3, 30, 13, 50, 6, 23, 0, 20, 12, 21, 8, 38],
            [9, 37, 10, 81, 11, 61, 14, 57, 8, 57, 0, 52, 7, 74, 6, 62, 12, 30, 1, 52, 2, 38, 13, 68, 4, 54, 3, 54, 5, 16]
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
# Initialize population with random schedules; each chromosome is a list of job tasks (job_id, machine, duration).
# Function to create a list of job tuples
def create_job_tuples(job_tasks):
    jobList = []
    for job_id, job in enumerate(job_tasks):
        for i in range(0, len(job), 2):
            jobList.append((job_id, job[i], job[i + 1]))
    return jobList

# Function to create a chromosome (a random permutation of tasks)
def create_chromosome(tasks):
    return random.sample(tasks, len(tasks))

# Function to create a population of chromosomes
def create_population(population_size, tasks, num_jobs):
    population = []
    while len(population) < population_size:
        chromosome = create_chromosome(tasks)
        if is_valid_chromosome(chromosome, num_jobs):
            population.append(chromosome)
    return population


# The fitness function evaluates a solution by calculating the makespan, which is the total time to complete all jobs on all machines.
def calculate_makespan(chromosome, num_machines):
    # Initialize completion times for machines and jobs
    machine_completion_times = [0] * num_machines
    job_completion_times = {}

    for task in chromosome:
        job_id, machine, duration = task

        # Initialize job completion time if not already done
        if job_id not in job_completion_times:
            job_completion_times[job_id] = 0

        # Determine the start time for the task
        start_time = max(machine_completion_times[machine], job_completion_times[job_id])

        # Update the completion time for the machine and the job
        machine_completion_times[machine] = start_time + duration
        job_completion_times[job_id] = start_time + duration

    # The makespan is the maximum completion time among all machines
    makespan = max(machine_completion_times)
    return makespan

# Function to calculate the fitness of a chromosome (using makespan directly)
def fitness_function(chromosome, num_machines):
    return calculate_makespan(chromosome, num_machines)

# Selection involves choosing the fittest individuals to create offspring using methods like roulette wheel and tournament selection.
def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(1 / f for f in fitness_scores)
    probabilities = [(1 / f) / total_fitness for f in fitness_scores]
    cumulative_probs = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]
    selected = []
    for _ in range(len(population)):
        r = random.random()
        for i, prob in enumerate(cumulative_probs):
            if r <= prob:
                selected.append(population[i])
                break
    return selected


# Tournament Selection
# This function selects two parents from the population based on tournament selection.
# A subset of individuals is randomly selected, and the best one from that subset is chosen.
def tournament_selection(population, num_machines, tournament_size=3):
    def select_one_parent():
        tournament = random.sample(population, tournament_size)
        fitness_scores = [(chromosome, fitness_function(chromosome, num_machines)) for chromosome in tournament]
        best_individual = min(fitness_scores, key=lambda x: x[1])[0]
        return best_individual

    parents = [select_one_parent(), select_one_parent()]
    return parents

# Crossover combines genetic information from two parents to create offspring using one-point or uniform crossover methods.
def one_point_crossover(parent1, parent2):
    if random.random() > 0.8:  # 80% crossover rate
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + [gene for gene in parent2 if gene not in parent1[:point]]
    child2 = parent2[:point] + [gene for gene in parent1 if gene not in parent2[:point]]
    return child1, child2

# Performs uniform crossover.
def uniform_crossover(parent1, parent2):
    if random.random() > 0.8:  # 80% crossover rate
        return parent1, parent2
    child1, child2 = [], []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(g1)
            child2.append(g2)
        else:
            child1.append(g2)
            child2.append(g1)
    return child1, child2
# Mutation: The process of introducing random changes in the offspring to maintain diversity.
# Mutates a chromosome by swapping two tasks.
def mutate(chromosome):
    if random.random() < 0.1:  # 10% mutation rate
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

# Uses inversion mutation.
def inversion_mutation(chromosome):
    if random.random() < 0.1:  # 10% mutation rate
        i, j = sorted(random.sample(range(len(chromosome)), 2))
        chromosome[i:j] = reversed(chromosome[i:j])
    return chromosome
# Validate the chromosome by checking if each job appears the once in the chromosome
def is_valid_chromosome(chromosome, num_jobs):
    job_counts = [0] * num_jobs
    for task in chromosome:
        job_id, machine, duration = task
        job_counts[job_id] += 1
    return all(count == len(chromosome) // num_jobs for count in job_counts)

# Combine all function into the genetic algorithm
def genetic_algorithm(datasets, num_generations, population_size, num_machines):
    # Initialize population
    job_tasks = datasets["la07"]["tasks"]
    num_jobs = datasets["la07"]["num_jobs"]
    tasks = create_job_tuples(job_tasks)
    population = create_population(population_size, tasks, num_jobs)

    # Evaluate the fitness of each individual in the population
    fitness_values = [fitness_function(chromosome, num_machines) for chromosome in population]

    best_fitness_values = []  # Initialize the list to store the best fitness values
    best_fitness = float('inf')
    best_chromosome = None
    stationary_count = 0

    for generation in range(num_generations):
        new_population = []

        # For each pair in the population
        for _ in range(population_size // 2):
            # Selection: select two individuals (c_a, c_b) from the population
            # Uncomment the desired selection method
            # parents = rank_selection(population, num_machines)
            # parents = roulette_wheel_selection(population, fitness_values)
            parents = tournament_selection(population, num_machines)

            # Crossover: (c_a', c_b') <- crossover of (c_a, c_b)
            # Uncomment the desired crossover method
            offspring1, offspring2 = one_point_crossover(parents[0], parents[1])
            # offspring1, offspring2 = uniform_crossover(parents[0], parents[1])

            # Mutation: mutate individuals (c_a', c_b')
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            # offspring1 = inversion_mutation(offspring1)
            # offspring2 = inversion_mutation(offspring2)

            # Validate offspring
            if is_valid_chromosome(offspring1, num_jobs):
                new_population.append(offspring1)
            if is_valid_chromosome(offspring2, num_jobs):
                new_population.append(offspring2)
        # Ensure the new population size is maintained
        while len(new_population) < population_size:
            chromosome = create_chromosome(tasks)
            if is_valid_chromosome(chromosome, num_jobs):
                new_population.append(chromosome)
        # Update population
        population = new_population

        # Evaluate the fitness of each individual in the new population
        fitness_values = [fitness_function(chromosome, num_machines) for chromosome in population]

        # Record the best fitness value of the current generation
        min_fitness = min(fitness_values)
        best_fitness_values.append(min_fitness)
        print(f"Generation {generation + 1}: Best Fitness (Makespan) = {min_fitness}")

        # Track the best chromosome
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_chromosome = population[fitness_values.index(min_fitness)]
            stationary_count = 0
        else:
            stationary_count += 1

        # Check for stationary state
        if stationary_count >= 800:  
            print("Reached stationary state.")
            break

    # Plot the evolution of the best fitness values
    plt.plot(range(1, len(best_fitness_values) + 1), best_fitness_values, marker='o')
    plt.title('Evolution of the Minimum Makespan')
    plt.xlabel('Generation')
    plt.ylabel('Minimum Makespan')
    plt.grid(True)
    plt.show()


# Example usage
num_generations = 5000
population_size = 10
num_machines = datasets["la07"]["num_machines"]
genetic_algorithm(datasets, num_generations, population_size, num_machines)
