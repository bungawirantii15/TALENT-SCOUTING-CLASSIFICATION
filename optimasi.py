import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class GeneticFeatureSelector:
    def __init__(self, X, y, population_size, generations, mutation_rate, crossover_rate, C, solver):
        self.X = X
        self.y = y
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
        model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
        model.fit(X_sel, self.y)
        pred = model.predict(X_sel)
        return accuracy_score(self.y, pred)
    
    # ------------------------------------------------------
    # 3. Seleksi: Roulette Wheel Selection
    # ------------------------------------------------------
    def selection(self, population, fitness_scores):
        if np.sum(fitness_scores) == 0:
            idx = np.random.choice(len(population), size=2, replace=False)
            return population[idx]

        probabilities = fitness_scores / np.sum(fitness_scores)
        idx = np.random.choice(len(population), size=2, replace=False, p=probabilities)
        return population[idx]

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
        best_chromosome = None
        best_score = 0
        
        for gen in range(1, self.generations + 1):
            fitness_scores = np.array([self.fitness(ind) for ind in population])
            best_idx = np.argmax(fitness_scores)

            if fitness_scores[best_idx] > best_score:
                best_score = fitness_scores[best_idx]
                best_chromosome = population[best_idx]

            print(f"Generasi {gen:02d} | Fitness terbaik saat ini: {best_score:.4f}")
            
            # ------------------------------------------------------
            # 🔥 ELITISME = 20% dari populasi
            # ------------------------------------------------------
            elitism_count = max(1, int(self.population_size * 0.20))
            top_idx = np.argsort(fitness_scores)[-elitism_count:]
            elite_individuals = [population[i].copy() for i in top_idx]

            # Children dimulai dari elit
            children = elite_individuals.copy()
            
            # ------------------------------------------------------
            # Seleksi parent & menghasilkan anak baru
            # ------------------------------------------------------
            while len(children) < self.population_size:
                parents = self.selection(population, fitness_scores)

                p1 = parents[np.random.randint(0, 2)]
                p2 = parents[np.random.randint(0, 2)]

                c1, c2 = self.crossover(p1, p2)

                children.append(self.mutate(c1))
                if len(children) < self.population_size:
                    children.append(self.mutate(c2))

            population = np.array(children)

        print("\nFitur terbaik ditemukan ✅")
        print("Subset fitur terbaik:\n", best_chromosome)
        print("Fitness terbaik:", best_score)

        return best_chromosome
            


        #     parents_idx = fitness_scores.argsort()[-2:]
        #     parent1, parent2 = population[parents_idx]

        #     new_population = []
        #     while len(new_population) < self.population_size:
        #         child1, child2 = self.crossover(parent1, parent2)
        #         new_population.append(self.mutate(child1))
        #         new_population.append(self.mutate(child2))

        #     population = np.array(new_population[:self.population_size])

        # final_fitness = np.array([self.fitness(ind) for ind in population])
        # best_idx = final_fitness.argmax()

        # print("Hasil Akhir GA")
        # print(f"Kromosom Terbaik : {population[best_idx]}")
        # print(f"Fitness Terbaik  : {final_fitness[best_idx]:.4f}")

        # return population[best_idx]


def train_hybrid_ga_lr(X_train, X_test, y_train, y_test, C_val, solver, pop, gen, mutation, crossover):
    ga = GeneticFeatureSelector(
        X_train, y_train,
        population_size=pop,
        generations=gen,
        mutation_rate=mutation,
        crossover_rate=crossover,
        C=C_val,
        solver = solver
    )

    best = ga.evolve()
    selected_features = X_train.columns[best == 1]

    model = LogisticRegression(C=C_val, solver=solver, max_iter=1000, random_state=42)
    model.fit(X_train[selected_features], y_train)

    pred = model.predict(X_test[selected_features])
    acc = accuracy_score(y_test, pred)

    return acc, pred, selected_features