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
# To initialize the population(a set of potential solutions), will need a list of random schedules for each job. 
# Meaning, the population is made by chromozomes, and each chromozome is a list of jobs.
# Each task represented as a touple (job_id, machine, duration), and a chromozone is a 
# sequence of these tasks: (0, 0, 47), (0, 4, 57), (0, 1, 71), (0, 3, 96), (0, 2, 14), (1, 0, 75), (1, 1, 60), (1, 4, 22), (1, 3, 79), (1, 2, 65), (2, 3, 32), (2, 0, 33), (2, 2, 69), (2, 1, 31), (2, 4, 58),...
# The order of tasks in the chromosome determines the sequence in which tasks are scheduled.
# This order affects how tasks are assigned to machines and their start times.
# The initial population consists of chromosomes with tasks in random orders.


# Initialize population p
# Evaluate the fitness of each individual in p
# For generation = 1 TO num_generations
#    P' = empty set
#    For pair = 1 TO population_size/2
#        Selection: select two individuals (c_a,c_b) from P
#        Crossover: (c_a',c_b') <- crossover of (c_a,c_b)
#        Mutation: mutate individuals (c_a', c_b')
#        P' <- P' U {c_a', c_b'}
#    End For
#    Elitism: add best fitted individual from P to P'
#    P <- P'
#    Evaluate fintness of all individuals in P
# End For

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
def create_population(population_size, tasks):
    return [create_chromosome(tasks) for _ in range(population_size)]

# Small test
jobList = create_job_tuples(datasets["abz5"]["tasks"])
population_size = 10
population = create_population(population_size, jobList)

# Print the population
for i, chromosome in enumerate(population):
    print(f"Chromosome {i+1}: {chromosome}")