import sys
import numpy as np


def predict(x, theta):
    """
    Le premier programme sera utilisé pour prédire le prix d’une voiture en fonction
    de son kilométrage. Quand vous lancerez le programme, celui ci vous demandera le
    kilométrage et devrait vous donner un prix approximatif de la voiture en utilisant
    l’hypothèse suivante :
    prixEstime(kilométrage) = θ0 + (θ1 * kilométrage)
    """
    return theta[0] + theta[1] * x


if __name__ == "__main__":
    # Reading data file
    try:
        theta = np.genfromtxt("model.csv", delimiter=',', skip_header=1)
    except:
        sys.exit("model.csv error")

    try:
        kilometers = float(
            input("kilométrage du véhicule : "))
    except:
        sys.exit("error")

    # Executing function and returning predicted value
    print(predict(kilometers, theta))
