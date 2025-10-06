import random
import matplotlib.pyplot as plt
import numpy as np

def load_cities(filename):
    cities = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:  # Only accept lines with 3 values
                name, x, y = parts
                cities.append([name, float(x), float(y)])
    return cities


def route_distance(route):
    dist = 0
    for i in range(len(route)):
        x1, y1 = route[i][1], route[i][2]
        x2, y2 = route[(i+1) % len(route)][1], route[(i+1) % len(route)][2]
        dist += ((x2-x1)**2 + (y2-y1)**2)**0.5
    return dist

def fitness(route):
    return 1 / route_distance(route)

def create_population(cities, size):
    population = []
    for i in range(size):
        temp = cities[:]
        random.shuffle(temp)
        population.append(temp)
    return population


def selection(population):
    population = sorted(population, key=lambda r: fitness(r), reverse=True)
    return population[:len(population)//2]


def crossover(parent1, parent2):
    start = random.randint(0, len(parent1)-2)
    end = random.randint(start+1, len(parent1)-1)
    child = parent1[start:end]
    for city in parent2:
        if city not in child:
            child.append(city)
    return child


def mutate(route, rate=0.01):
    for i in range(len(route)):
        if random.random() < rate:
            j = random.randint(0, len(route)-1)
            route[i], route[j] = route[j], route[i]
    return route


def next_generation(population, mutation_rate=0.01):
    selected = selection(population)
    children = []
    for i in range(len(population)):
        p1 = random.choice(selected)
        p2 = random.choice(selected)
        child = crossover(p1, p2)
        child = mutate(child, mutation_rate)
        children.append(child)
    return children


def plot_route(route, title):
    x = [c[1] for c in route] + [route[0][1]]
    y = [c[2] for c in route] + [route[0][2]]
    plt.plot(x, y, 'o-r')
    for c in route:
        plt.text(c[1], c[2], c[0])
    plt.title(title)
    plt.show()


def genetic_algorithm(filename, population_size=50, generations=100, mutation_rate=0.02):
    cities = load_cities(filename)
    population = create_population(cities, population_size)
    avg_fitness = []

    for g in range(generations):
        population = next_generation(population, mutation_rate)
        best = sorted(population, key=lambda r: fitness(r), reverse=True)[0]
        avg_fit = np.mean([fitness(r) for r in population])
        avg_fitness.append(avg_fit)

        
        if g % 20 == 0 or g == generations-1:
            print("Generation:", g, "Best Distance:", route_distance(best))
            plot_route(best, "Best Route at Generation " + str(g))

    
    plt.plot(avg_fitness)
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness (1/Distance)")
    plt.title("Average Fitness over Generations")
    plt.show()

    return best

if __name__ == "__main__":
    best_route = genetic_algorithm("India_cities_GA.txt", population_size=50, generations=100)
    print("\nFinal Best Route:")
    for c in best_route:
        print(c[0], end=" -> ")
    print(best_route[0][0])

