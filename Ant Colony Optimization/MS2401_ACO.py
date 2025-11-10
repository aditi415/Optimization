import numpy as np
import matplotlib.pyplot as plt
import random
import math

def load_cities(filename="India_cities.txt"):
    cities = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) == 3:
                name, lat, lon = parts
                cities.append((name, float(lat), float(lon)))
    return cities

def distance_matrix(cities):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371 
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = cities[i][1], cities[i][2]
                lat2, lon2 = cities[j][1], cities[j][2]
                dist[i][j] = haversine(lat1, lon1, lat2, lon2)
            else:
                dist[i][j] = 0
    return dist
    
class ACO_TSP:
    def __init__(self, dist_matrix, n_ants=20, alpha=1, beta=5, rho=0.5, Q=100, iterations=60):
        self.dist = dist_matrix
        self.n = len(dist_matrix)
        self.n_ants = n_ants
        self.alpha = alpha    
        self.beta = beta       
        self.rho = rho          
        self.Q = Q              
        self.iterations = iterations
        self.pheromone = np.ones((self.n, self.n))
        self.best_distance = float("inf")
        self.best_tour = None

    def run(self, cities):
        for iteration in range(1, self.iterations + 1):
            all_tours = []
            for _ in range(self.n_ants):
                tour = self.construct_tour()
                all_tours.append(tour)

            self.update_pheromones(all_tours)
            best_tour, best_dist = self.find_best_tour(all_tours)

            if best_dist < self.best_distance:
                self.best_distance = best_dist
                self.best_tour = best_tour

            if iteration % 10 == 0 or iteration == 1:
                print(f"Iteration {iteration:02d} | Current Best Distance: {self.best_distance:.2f}")
                self.plot_path(cities, self.best_tour, iteration)

        return self.best_tour, self.best_distance

    def construct_tour(self):
        start = random.randint(0, self.n - 1)
        tour = [start]
        unvisited = set(range(self.n))
        unvisited.remove(start)

        while unvisited:
            current = tour[-1]
            next_city = self.select_next_city(current, unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)
        return tour

    def select_next_city(self, current, unvisited):
        unvisited = list(unvisited)
        pheromone = self.pheromone[current, unvisited] ** self.alpha
        heuristic = (1 / self.dist[current, unvisited]) ** self.beta
        probabilities = pheromone * heuristic
        probabilities /= probabilities.sum()
        return random.choices(unvisited, weights=probabilities)[0]

    def update_pheromones(self, tours):
        self.pheromone *= (1 - self.rho)
        for tour in tours:
            distance = self.tour_length(tour)
            contribution = self.Q / distance
            for i in range(len(tour) - 1):
                a, b = tour[i], tour[i + 1]
                self.pheromone[a][b] += contribution
                self.pheromone[b][a] += contribution

    def find_best_tour(self, tours):
        best_tour = None
        best_dist = float("inf")
        for t in tours:
            d = self.tour_length(t)
            if d < best_dist:
                best_tour, best_dist = t, d
        return best_tour, best_dist

    def tour_length(self, tour):
        length = 0
        for i in range(len(tour) - 1):
            length += self.dist[tour[i]][tour[i + 1]]
        length += self.dist[tour[-1]][tour[0]]  
        return length

    def plot_path(self, cities, tour, iteration):
        plt.figure(figsize=(8, 7))
        for i in range(len(tour)):
            c1 = cities[tour[i]]
            c2 = cities[tour[(i + 1) % len(tour)]]
            plt.plot([c1[2], c2[2]], [c1[1], c2[1]], 'b-', linewidth=1.5, alpha=0.7)

        for (name, lat, lon) in cities:
            plt.scatter(lon, lat, color='red', s=60)
            plt.text(lon + 0.1, lat + 0.1, name, fontsize=9, fontweight='bold')

        plt.title(f"ACO Iteration {iteration} - Best Path", fontsize=12)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"aco_tsp_iteration_{iteration}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    cities = load_cities("India_cities.txt")
    if not cities:
        print("No cities loaded! Please check 'India_cities.txt' format.")
        exit()

    dist = distance_matrix(cities)
    aco = ACO_TSP(dist_matrix=dist, n_ants=15, iterations=60)
    best_tour, best_distance = aco.run(cities)

    print("\nFinal Best Tour (in visiting order):")
    for i, idx in enumerate(best_tour, start=1):
        print(f"{i}. {cities[idx][0]} ({cities[idx][1]}, {cities[idx][2]})")

    print(f"\nTotal Distance (round trip): {best_distance:.2f} km")

    aco.plot_path(cities, best_tour, "Final")
