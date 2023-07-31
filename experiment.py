import random

import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import seaborn as sns
from teeplot import teeplot as tp
import tqdm

from evolve import evolve, toolbox
from robustness import test_robustness


def sample_robustness(npop, ngen=100):

    population = evolve(npop, ngen)

    # Select a random individual from the final population
    random_individual = random.choice(population)

    # Test the robustness of the random individual
    return test_robustness(
        random_individual,
        toolbox,
        toolbox.mutate,
    )

def robustness_samples(pop_sizes, n_samples, ngen=100):
    robustness_results = []

    for npop in pop_sizes:
        for _ in tqdm.tqdm(range(n_samples), desc=f"Sampling npop={npop}"):
            robustness_results.append((npop, sample_robustness(npop, ngen)))

    return robustness_results

# Define the population sizes and number of samples
pop_sizes = [10, 1000]
n_samples = 20

# Sample robustness and plot results
results = robustness_samples(pop_sizes, n_samples)

# Transform results into a DataFrame for easier plotting
import pandas as pd

df = pd.DataFrame(results, columns=["Population Size", "Robustness"])

# Assuming your "Population Size" column contains discrete categories, not continuous numbers.
unique_population_sizes = df["Population Size"].unique()

# Generate a list of series, each series representing the "Robustness" values for a unique population size
groups = [df[df["Population Size"] == pop_size]["Robustness"] for pop_size in unique_population_sizes]

# Perform the Kruskal-Wallis H test
stat, p = scipy_stats.kruskal(*groups)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# Interpretation
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')

# Plot using seaborn
plt.figure(figsize=(10, 6))
tp.tee(
    sns.boxplot,
    data=df,
    x="Population Size",
    y="Robustness",
    showfliers=False,
    notch=True,
)
plt.title("Robustness by Population Size")

plt.show()
