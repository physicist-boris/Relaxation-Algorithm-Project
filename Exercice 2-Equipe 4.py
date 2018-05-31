from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product, combinations

import plotly.plotly as py
import plotly.graph_objs as go

# On se crée une matrice 3D sur python où chaque entrée de la matrice correspond à la valeur du potentiel à un noeud
# de notre cube 3D de 10cm de côté

pv_matrix = np.zeros((51, 51, 51))

# On fixe les conditions initiales dans la matrice: On donne des valeurs de potentiel aux noeuds libres de la matrice 3D

pv_matrix[:, :, 50] = -100  # droite
pv_matrix[0, :, :] = -150  # derriere
pv_matrix[50, :, :] = 0 # devant
pv_matrix[:, :, 0] = 100  # gauche
pv_matrix[:, 50, :] = 200  # dessous
pv_matrix[:, 0, :] = 0  # dessus


list_variation = []


def relaxation_method(critère_arrêt, matrix_iter):
    """
    Fonction qui applique la méthode de relaxation vue en cours avec un critère d'arrêt prédéterminé
    et qui retourne le nombre d'itérations nécessaires pour atteindre ce critère

    :param critère_arrêt:
    :param matrix_iter: matrice à itérer
    :return: iter_number(nombres d'itérations)
    """
    max_variation = 10
    iter_number = 0
    while max_variation > critère_arrêt:
        initial_matrix = matrix_iter.copy()
        matrix_iter[1:-1, 1:-1, 1:-1] = (matrix_iter[:-2, 1:-1, 1:-1] + matrix_iter[2:, 1:-1, 1:-1]
                                         + matrix_iter[1:-1, :-2, 1:-1] + matrix_iter[1:-1, 2:, 1:-1]
                                         + matrix_iter[1:-1, 1:-1, :-2] + matrix_iter[1:-1, 1:-1, 2:]) / 6

        variation_matrix = matrix_iter - initial_matrix

        # max_variation représente la plus grande variation qu'un noeud du cube ait subit, entre deux itérations,
        # parmis toutes les variations des différents noeuds:

        max_variation = max([abs(np.max(variation_matrix)), abs(np.min(variation_matrix))])
        list_variation.append(max_variation)
        iter_number += 1

    return iter_number


nombres_iter = (relaxation_method(10 ** (-6), pv_matrix))

fig = plt.figure(1)


def créer_graphique(subplot_number, matrix_part, axe_a, axe_b, axe_c):
    """
    Fonction génère un graphique 2D en couleur du potentiel pour une face de la matrice 3D

    :param subplot_number: numéro correpondant au subplot du graphique
    :param matrix_part: partie 2D, de la matrice 3D, à tracer
    :param axe_a: axe x, y, ou z (sous forme de chaînes de caractères)
    :param axe_b: axe x, y, ou z (sous forme de chaînes de caractères)
    :param axe_c: axe x, y, ou z (sous forme de chaînes de caractères)
    :return:
    """
    ax = plt.subplot(subplot_number)
    ax.set_title('Potentiel dans le plan {}-{} \n en {}= 0cm'.format(axe_a, axe_b, axe_c))
    ax.set_xlabel("{} (cm)".format(axe_a))
    ax.set_ylabel("{} en (cm)".format(axe_b))
    ax.axis("equal")

    im = plt.imshow(matrix_part, cmap="jet", extent=[-5, 5, -5, 5])

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)


créer_graphique(231, pv_matrix[25, :, :], "x", "y", "z")
créer_graphique(232, pv_matrix[:, ::-1, 25], "y", "z", "x")
créer_graphique(233, pv_matrix[:, 25, :], "x","z", "y")

# Tracer le graphique de convergence
fig2 = plt.figure(2)
ax1 = plt.subplot(111)
ax1.set_title('Graphique de convergence')
ax1.set_xlabel("nombres d'itérations")
ax1.set_ylabel("Variation")

ax1.set_yscale("log")

x1 = np.linspace(0, nombres_iter, num=nombres_iter)
y1 = np.array(list_variation)
plt.plot(x1, y1, "r")

# Courte justification du critère d'arrêt:
# Ici on a choisit 10^(-6) V comme critère d'arrêt qui représente que la plus grande variation de potentiel d'un
# noeud du cube entre deux itérations ne doit pas plus grande que cette valeur de telle sorte que une variation
# de potentiel dans le cube est à peine perceptible au-dela de cette valeur(après un certain nombre d'itération),
# on dit alors que l'on a atteint un équilibre


plt.tight_layout()

plt.show()

# Potentiel au centre du cube (0,0,0): 8.33 V

print("Potentiel au centre du cube:{} V".format(pv_matrix[25, 25, 25]))
