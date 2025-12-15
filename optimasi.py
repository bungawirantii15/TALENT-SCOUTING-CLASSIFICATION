import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class GeneticFeatureSelector:
    def __init__(self, X, y, X_test, y_test, population_size, generations, mutation_rate, crossover_rate, C, solver):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.C = C
        self.solver = solver    
        self.n_features = X.shape[1]

    def init_population(self):
        return np.random.randint(0, 2, size=(self.population_size, self.n_features))

    def fitness(self, chromosome):
        if chromosome.sum() == 0:
            return 0
        X_sel = self.X.iloc[:, chromosome == 1]
        X_test_sel = self.X_test.iloc[:, chromosome == 1]
        model = LogisticRegression(max_iter=300, solver=self.solver, C=self.C, random_state=42)
        model.fit(X_sel, self.y)
        pred = model.predict(X_test_sel)
        return accuracy_score(self.y_test, pred)

    def mutate(self, chromosome):
        for i in range(self.n_features):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, self.n_features - 1)
        child1 = np.append(parent1[:point], parent2[point:])
        child2 = np.append(parent2[:point], parent1[point:])
        return child1, child2

    def evolve(self):
        population = self.init_population()
        for gen in range(self.generations):
            fitness_scores = np.array([self.fitness(ind) for ind in population])
            best_idx = fitness_scores.argmax()
            best_fit = fitness_scores[best_idx]

            print(f"Generasi {gen + 1} -> Fitness Terbaik  : {best_fit:.4f}")

            parents_idx = fitness_scores.argsort()[-2:]
            parent1, parent2 = population[parents_idx]

            new_population = []
            while len(new_population) < self.population_size:
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))

            population = np.array(new_population[:self.population_size])

        final_fitness = np.array([self.fitness(ind) for ind in population])
        best_idx = final_fitness.argmax()

        print("Hasil Akhir GA")
        print(f"Kromosom Terbaik : {population[best_idx]}")
        print(f"Fitness Terbaik  : {final_fitness[best_idx]:.4f}")

        return population[best_idx]


def train_hybrid_ga_lr(X_train, X_test, y_train, y_test, C_val, solver, pop, gen, mutation, crossover):
    ga = GeneticFeatureSelector(
        X_train, y_train,
        X_test, y_test,
        population_size=pop,
        generations=gen,
        mutation_rate=mutation,
        crossover_rate=crossover,
        C=C_val,
        solver = solver
    )

    best = ga.evolve()
    selected_features = X_train.columns[best == 1]

    model = LogisticRegression(C=C_val, solver=solver, max_iter=300, random_state=42)
    model.fit(X_train[selected_features], y_train)

    pred = model.predict(X_test[selected_features])
    acc = accuracy_score(y_test, pred)

    return acc, pred, selected_features