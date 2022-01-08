#örnek rota probleminin uygulanması
import numpy as np

from ant_colony import AntColony

distances = np.array([[np.inf, 2, 2, 5, 7], 
                      [2, np.inf, 4, 8, 2],
                      [2, 4, np.inf, 1, 3],
                      [5, 8, 1, np.inf, 2],
                      [7, 2, 3, 2, np.inf]])

ant_colony = AntColony(distances, 10, 1, 500, 0.95, alpha=0.05, beta=1)
shortest_path = ant_colony.run()
print ("En kısa yol: {}".format(shortest_path))
