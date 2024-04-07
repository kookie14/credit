import random

class NeighborhoodGenerator:
    def __init__(self):
        self.neighbor_generators = {
            'N1': self._generate_N1,
            'N2': self._generate_N_sequential(2),
            'N3': self._generate_N_sequential(3),
            'N4': self._generate_N_random(1),
            'N5': self._generate_N_random(2),
            'N6': self._generate_N_random(3),
            'N7': self._generate_N_random(4),
            'N8': self._generate_N_random(5),
        }

    def generate_neighbor(self, current_solution, neighbor_type):
        if neighbor_type in self.neighbor_generators:
            return self.neighbor_generators[neighbor_type](current_solution)
        else:
            raise ValueError(f"Unknown neighbor type: {neighbor_type}")

    def _generate_N1(self, current_solution):
        neighbor_solutions = []
        for i in range(len(current_solution)):
            neighbor = list(current_solution)
            neighbor[i] = 1 - neighbor[i]
            neighbor_solutions.append(neighbor)
        return neighbor_solutions

    def _generate_N_sequential(self, num_indices):
        def generate_sequential(current_solution):
            neighbor_solutions = []
            for i in range(len(current_solution) - num_indices + 1):
                neighbor = list(current_solution)
                for j in range(num_indices):
                    neighbor[i + j] = 1 - neighbor[i + j]
                neighbor_solutions.append(neighbor)
            return neighbor_solutions
        return generate_sequential

    def _generate_N_random(self, num_indices):
        def generate_random(current_solution):
            neighbor_solutions = []
            for _ in range(num_indices): # Generate num_indices number of neighbor solutions
                neighbor = list(current_solution)
                indices_to_change = random.sample(range(len(neighbor)), num_indices)
                for index in indices_to_change:
                    neighbor[index] = 1 - neighbor[index]
                neighbor_solutions.append(neighbor)
            return neighbor_solutions
        return generate_random
    
def main():
    neighborhood_generator = NeighborhoodGenerator()
    current_solution = [1,1,1,1,1,1,1,1]
    neighbors_N1 = neighborhood_generator.generate_neighbor(current_solution, 'N1')
    print("Neighbors of type N1:")
    for neighbor in neighbors_N1:
        print(neighbor)

    neighbors_N2 = neighborhood_generator.generate_neighbor(current_solution, 'N2')
    print("\nNeighbors of type N2:")
    for neighbor in neighbors_N2:
        print(neighbor)

    neighbors_N3 = neighborhood_generator.generate_neighbor(current_solution, 'N3')
    print("\nNeighbors of type N3:")
    for neighbor in neighbors_N3:
        print(neighbor)

    # Generate neighbors of type N4 (randomly flip 1 bit)
    neighbor_N4 = neighborhood_generator.generate_neighbor(current_solution, 'N4')
    print("\nNeighbor of type N4:")
    print(neighbor_N4)

    # Generate neighbors of type N5 (randomly flip 2 bits)
    neighbor_N5 = neighborhood_generator.generate_neighbor(current_solution, 'N5')

    print("\nNeighbor of type N5:")
    print(neighbor_N5)

    # Generate neighbors of type N6 (randomly flip 3 bits)
    neighbor_N6 = neighborhood_generator.generate_neighbor(current_solution, 'N6')
    print("\nNeighbor of type N6:")
    print(neighbor_N6)

    # Generate neighbors of type N7 (randomly flip 4 bits)
    neighbor_N7 = neighborhood_generator.generate_neighbor(current_solution, 'N7')
    print("\nNeighbor of type N7:")
    print(neighbor_N7)

    # Generate neighbors of type N8 (randomly flip 5 bits)
    neighbor_N8 = neighborhood_generator.generate_neighbor(current_solution, 'N8')
    print("\nNeighbor of type N8:")
    print(neighbor_N8)

if __name__ == "__main__":
    main()
