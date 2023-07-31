from deap import base, creator, tools, algorithms
import numpy as np
import random
from typing import List, Tuple
import math

from grn import evaluate_grn

# Define constants
GENE_COUNT = 3
POPULATION_SIZE = 100
GENERATIONS = 50
CROSSOVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.2
SIGMA_SHARE = 0.5  # This is the sigma parameter in the sharing function

# Initialize DEAP tools
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # We only want to minimize variance now
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Define how to create an individual (a GRN) and a population
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=GENE_COUNT**2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the evaluation function
def evaluate(individual: creator.Individual) -> Tuple[float,]:
    grn = np.array(individual).reshape((GENE_COUNT, GENE_COUNT))

    # We can use the evaluate_grn function defined earlier here
    expression = evaluate_grn(grn, np.ones(GENE_COUNT))

    # We only want to minimize the variance among expression values
    return np.var(expression)

# Define the fitness sharing function
# def fitness_sharing(
#     individual: creator.Individual,
#     population: List[creator.Individual],
#     sigma: float,
# ) -> float:
#     distances = np.array([math.sqrt(np.sum((np.array(individual) - np.array(other))**2)) for other in population])
#     sharing = np.sum([1 - (dist / sigma) if dist < sigma else 0 for dist in distances])
#     return individual.fitness.values[0] / sharing

# Define the genetic operations
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the genetic algorithm with fitness sharing
def evolve_(
    population: List[creator.Individual],
    toolbox: base.Toolbox,
    cxpb: float,
    mutpb: float,
    ngen: int,
    # sigma_share: float,
) -> List[creator.Individual]:
    # Evaluate the first generation
    fits = [toolbox.evaluate(ind) for ind in population]
    for ind, fit in zip(population, fits):
        ind.fitness.values = fit,

    for gen in range(ngen):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with sharing
        fits = [toolbox.evaluate(ind) for ind in offspring]
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit,

        # fits_shared = [
        #     fitness_sharing(ind, offspring, sigma_share) for ind in offspring
        # ]
        # for ind, fit in zip(offspring, fits_shared):
        #     ind.fitness.values = fit,

        # Replace the current population by the offspring
        population[:] = offspring

    return population

def evolve(
    npop: int,
    ngen: int,
) -> List[creator.Individual]:
    population = toolbox.population(n=npop)
    return evolve_(
        population,
        toolbox,
        CROSSOVER_PROBABILITY,
        MUTATION_PROBABILITY,
        ngen,
    )

if __name__ == "__main__":
    population = evolve(
        POPULATION_SIZE,
        GENERATIONS,
    )

    best_individual = tools.selBest(population, k=1)[0]
    print('Best GRN:', np.array(best_individual).reshape((GENE_COUNT, GENE_COUNT)))
    print('Best Fitness:', best_individual.fitness.values)
