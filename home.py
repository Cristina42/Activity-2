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



# Uses roulette wheel selection.
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

# Performs one-point crossover.
def one_point_crossover(parent1, parent2):
    if random.random() > 0.8:  # 80% crossover rate
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + [gene for gene in parent2 if gene not in parent1[:point]]
    child2 = parent2[:point] + [gene for gene in parent1 if gene not in parent2[:point]]
    return child1, child2

# Performs uniform crossover.
def uniform_crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
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


# Mutates a chromosome by swapping two tasks.
def mutate(chromosome):
    if random.random() < 0.1:  # 10% mutation rate
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

# Uses inversion mutation.
def inversion_mutation(chromosome):
    if random.random() < MUTATION_RATE:
        i, j = sorted(random.sample(range(len(chromosome)), 2))
        chromosome[i:j] = reversed(chromosome[i:j])
    return chromosome


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

        # Selection: Choose between techniques
        selected = select_parents(pop, fitnesses)  # Default: Tournament
        # selected = roulette_wheel_selection(pop, fitnesses)  # Uncomment to use Roulette Wheel

        next_pop = []
        for i in range(0, len(selected), 2):
            p1, p2 = selected[i], selected[(i + 1) % len(selected)]
            
            # Crossover: Choose between techniques
            c1, c2 = one_point_crossover(p1, p2)  # Default: One-Point
            # c1, c2 = uniform_crossover(p1, p2)  # Uncomment to use Uniform Crossover

            # Mutation: Apply chosen technique
            next_pop.append(mutate(c1))  # Default: Swap Mutation
            next_pop.append(mutate(c2))  # Default: Swap Mutation
            # next_pop.append(inversion_mutation(c1))  # Uncomment to use Inversion Mutation
            # next_pop.append(inversion_mutation(c2))  # Uncomment to use Inversion Mutation
        
        pop = next_pop
    return best_chromosome, best_fitness

# Show the Gantt chart of the best schedule.
def plot_gantt_chart(schedule, num_machines):
    machine_time = [0] * num_machines
    colors = {}
    plt.figure(figsize=(10, 6))

    for job_id, machine, duration in schedule:
        start_time = machine_time[machine]
        plt.barh(machine, duration, left=start_time, color=colors.setdefault(job_id, f"C{job_id % 10}"), edgecolor="black")
        plt.text(start_time + duration / 2, machine, f"job{job_id}", ha="center", va="center", color="white", fontsize=8)
        machine_time[machine] += duration

    plt.xlabel("Time")
    plt.ylabel("Machines")
    plt.yticks(range(num_machines), [f"Machine {i}" for i in range(num_machines)])
    plt.title("Gantt Chart")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for dataset_name, dataset in datasets.items():
        num_jobs = dataset["num_jobs"]
        num_machines = dataset["num_machines"]
        job_tasks = dataset["tasks"]
        best_schedule, best_makespan = genetic_algorithm()

        print(f"\nDataset: {dataset_name}\nOptimal Makespan: {best_makespan}")
        plot_gantt_chart(best_schedule, num_machines)

