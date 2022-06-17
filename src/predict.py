import sys
import numpy as np


def predict_without_train(x):
    return 0 + 0 * x

def predict(x, theta):
    """
    Le premier programme sera utilisé pour prédire le prix d’une voiture en fonction
    de son kilométrage. Quand vous lancerez le programme, celui ci vous demandera le
    kilométrage et devrait vous donner un prix approximatif de la voiture en utilisant
    l’hypothèse suivante :
    prixEstime(kilométrage) = θ0 + (θ1 * kilométrage)
    """
    return theta[0] + theta[1] * x


def get_data():
    try:
        theta = np.genfromtxt("model.csv", delimiter=',', skip_header=1)
    except:
        try:
            klm = float(
            input("kilométrage du véhicule : "))
        except:
            sys.exit("error")
        print(predict_without_train(klm))
        sys.exit("use train before")
    try:
        klm = float(
            input("kilométrage du véhicule : "))
    except:
        sys.exit("error")
    print(predict(klm, theta))


if __name__ == "__main__":
    get_data()
