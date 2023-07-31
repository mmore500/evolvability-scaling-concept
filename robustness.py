import random

from deap import base, creator, tools, algorithms
import numpy as np

from evolve import evolve, toolbox

def test_robustness(
    individual: creator.Individual,
    toolbox: base.Toolbox,
    mutation_operator: callable,
    trials: int = 1000,
) -> float:
    robustness_values = []

    for _ in range(trials):
        # Clone the individual to not alter the original
        cloned_individual = toolbox.clone(individual)

        # Apply mutation
        mutation_operator(cloned_individual)

        # Evaluate the fitness of the original and the mutated individual
        original_fitness = toolbox.evaluate(individual)
        mutated_fitness = toolbox.evaluate(cloned_individual)

        # Calculate the robustness as the absolute difference between the fitnesses
        robustness = abs(original_fitness - mutated_fitness)
        robustness_values.append(robustness)

    # Calculate the average robustness over the trials
    average_robustness = sum(robustness_values) / trials

    return average_robustness


if __name__ == "__main__":

    population = evolve(100, 100)

    # Select a random individual from the final population
    random_individual = random.choice(population)
    print('Random GRN:', random_individual)
    print('Fitness:', random_individual.fitness.values)

    # Test the robustness of the random individual
    robustness = test_robustness(
        random_individual,
        toolbox,
        toolbox.mutate,
    )
    print('Robustness:', robustness)
