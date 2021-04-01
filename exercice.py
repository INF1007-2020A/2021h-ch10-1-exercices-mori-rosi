#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:

    #return np.arange(start=-1.3, stop=2.5, step=(2.5+1.3)/64) # pas obligé d'écrire start, stop, step si on respecte l'ordre
    return np.linspace(start=-1.3, stop=2.5, num=64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:

    ## brouillon
    # polar_coordinates = []
    # for i in range(len(cartesian_coordinates)):
    #     x = cartesian_coordinates[i][0]
    #     y = cartesian_coordinates[i][1]
    #
    #     polar_coordinates.append(((x**2 + y**2)**0.5, np.arctan(y/x)))
    #
    # np.array(polar_coordinates)
    #
    # return polar_coordinates

    ##solution perso
    polar_coordinates = []
    for coord in cartesian_coordinates:
        x = coord[0]
        y = coord[1]

        polar_coordinates.append((np.sqrt(x**2+y**2), np.arctan(y/ x)))

    return np.array(polar_coordinates)


    ##corrigé 2:compréhension de liste
    #return np.array([(np.sqrt(coord[0] ** 2 + coord[1] ** 2), np.arctan(coord[1] / coord[0])) for coord in cartesian_coordinates])

    ##corrigé 1
    #
    # coord_polair = []
    # for coord in cartesian_coordinates: # plus rapide en termes de temps d'exécution
    #     x = coord[0]
    #     y = coord[1]
    #     rayon = np.sqrt(x**2 + y**2)
    #     angle = np.arctan(y/x) # ou angle = np.arctan2(y, x)
    #     coord_polair.append((rayon, angle))
    #
    # coord_polair = np.array(coord_polair)
    #
    # return coord_polair


def find_closest_index(values: np.ndarray, number: float) -> int:

 #Calculer la différence -> array des différenes
 #Caster l'array des différences en valeur absolue
 #trouver l'index de la valeur la plus petite avec argmin()

    return (abs(values - number)).argmin()


def create_plot():
    x = np.linspace(-1, 1, 250)
    y = x**2 * np.sin(1/x**2) + x

    plt.scatter(x, y, c='crimson', s=3**2, label="courbe1")
    plt.title("Graph de y exo 10.1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def monte_carlo(iteration: int) -> float:
    #Aire du quart de cercle de rayon 1 => pi/4

    x_inside_circle = []
    x_outside_circle = []
    y_inside_circle = []
    y_outside_circle = []

    for i in range(iteration):
        x = np.random.random() #génère chiffre float entre 0 et 1 -> coordonnée en x quelconque
        y = np.random.random()

        if np.sqrt(x**2+y**2) <= 1.0:
            x_inside_circle.append(x)
            y_inside_circle.append(y)

        else:
            x_outside_circle.append(x)
            y_outside_circle.append(y)

    plt.scatter(x_inside_circle, y_inside_circle, label="inside circle", s=5, c="crimson")
    plt.scatter(x_outside_circle, y_outside_circle, label="outside circle", s=1, c="b")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Estimer pi avec la méthode de Monte Carlo")
    plt.legend()
    plt.show()

    return (len(x_inside_circle) / iteration) * 4 ##**Pourquoi??
    # / nb de points

def monte_carlo_eff(iteration: int) -> float:

    x = np.random.rand(iteration) #crée array de valeurs float aléatoires entre 0 et 1
    y = np.random.rand(iteration)
    rayon = np.sqrt(x**2+y**2)
    pi = np.count_nonzero(rayon <= 1) / iteration * 4
    return pi


def integrand() -> tuple:

    I = quad(lambda x: np.exp(-x**2), -np.inf, np.inf)

    x = np.arange(-4, 4, 0.1)
    y = [quad(lambda x: np.exp(-x**2), 0, value)[0] for value in x]

    plt.plot(x, y, c="crimson")
    plt.xlim(-4, 4)
    plt.ylim(-1, 2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plot Integral")
    plt.show()

    return I



#def integrate(y):




if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici

    #rand_array = linear_values()
    #print(rand_array)

    #print(coordinate_conversion(cartesian_coordinates=np.array([(15, 30), (-7, 10)])))

    #create_plot()

    #print(monte_carlo(10000))
    #print(monte_carlo_eff(10000))

    print(integrand())
    #print(integrate(y))
