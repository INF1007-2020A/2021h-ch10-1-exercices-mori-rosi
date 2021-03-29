#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:

    return np.arange(start=-1.3, stop=2.5, step=(2.5+1.3)/64) # pas obligé de d'écrire start, stop, step si on respecte l'ordre
    #return np.linspace(start=-1.3, stop=2.5, num=64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:

    for i in range(len(cartesian_coordinates)):
        x = cartesian_coordinates[i][0]
        y = cartesian_coordinates[i][1]

        polar_coordinates = np.array([(x**2 + y**2)**0.5, np.arctan(y/x)])

    return polar_coordinates


def find_closest_index(values: np.ndarray, number: float) -> int:
    return 0


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici

    rand_array = linear_values()
    print(rand_array)

    print(coordinate_conversion(cartesian_coordinates=np.array([[15, 30], [-7, 10]])))
