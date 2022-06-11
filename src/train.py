import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd


def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    Returns:
    x as a numpy.array, a vector of shape m * 2.
    None if x is not a numpy.array.
    None if x is a empty numpy.array.
    Raises:
    This function should not raise any Exception"""
    if not isinstance(x, np.ndarray):
        return None
    try:
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        i = np.ones((x.shape[0], 1))
        return np.append(i, x, axis=1)
    except ValueError:
        return None


def loss_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    try:
        return (y_hat - y) ** 2 / (2 * y.shape[0]) * 10
    except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
        return None


def loss_(y: np.ndarray, y_hat: np.ndarray) -> float:
    if loss_elem_(y, y_hat) is None:
        return None
    try:
        return np.sum(loss_elem_(y, y_hat)) / 10
    except (np.core._exceptions.UFuncTypeError, TypeError):
        return None


def predict_(x, theta):
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if add_intercept(x).shape[1] != theta.shape[0]:
        return None
    return np.dot(add_intercept(x), theta)


def gradient(x: np.ndarray, y: np.ndarray, theta) -> None:
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be a numpy.array, a matrix of shape m * 1.
    y: has to be a numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta is an empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception."""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    try:
        m = x.shape[0]
        x = add_intercept(x)
        res = x.T.dot(x.dot(theta) - y)
    except (np.core._exceptions.UFuncTypeError, TypeError, ValueError):
        return None
    return res / m


def fit_(x: np.ndarray, y: np.ndarray, theta, alpha, max_iter) -> None:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray) or\
            not isinstance(alpha, float) or not isinstance(max_iter, int):
        return None
    if max_iter <= 0:
        return None
    for i in range(max_iter):
        swp = gradient(x, y, theta)
        tmp = (swp * alpha)
        theta = theta - tmp
    return theta


def draw_regression(x, y, theta):
    fx = [min(x), max(x)]
    fy = [0, 0]
    fy[0] = theta[1] * fx[0] + theta[0]
    fy[1] = theta[1] * fx[1] + theta[0]

    y_hat = predict_(x, theta)
    for x_i, y_hat_i, y_i in zip(x, y_hat, y):
        plt.plot([x_i, x_i], [y_i, y_hat_i], 'r--')
    plt.plot(x, y, "o")
    plt.plot(fx, fy, "-b")
    plt.xlabel("Klm")
    plt.ylabel("Prix")
    plt.show()


def normalize(array):
    ret = np.empty([])
    for elem in array:
        ret = np.append(ret, (elem - min(array)) / (max(array) - min(array)))
    return (ret[1:])


def denormalized_theta(miles, price, theta):
    fx = [min(miles), max(miles)]
    fy = []
    for elem in fx:
        elem = theta[1] * ((elem - fx[0]) / (fx[1] - fx[0])) + theta[0]
        fy.append((elem * (max(price) - min(price))) + min(price))
    a = (fy[0] - fy[1]) / (fx[0] - fx[1])
    b = fy[0] - (a * fx[0])
    return (np.array([b, a]))


def save_model(theta):
    try:
        np.savetxt("model.csv", theta.reshape(1, -1), delimiter=',',
                   header="theta0,theta1", fmt="%1.8f")
    except:
        sys.exit("Error saving model")


def read_csv_file():
    if len(sys.argv) != 2:
        sys.exit("ft_linear_regression: Wrong argument number")
    try:
        data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
    except:
        sys.exit("ft_linear_regression: Unable to open file")
    print(data[:, 0])
    print(data[:, 1])

    print(normalize(data[:, 0]))
    print(normalize(data[:, 1])) 
    
    return (data[:, 0], data[:, 1], normalize(data[:, 0]), normalize(data[:, 1]))


if __name__ == "__main__":
    miles, price, x, y = read_csv_file()
    theta = fit_(x, y, np.array([1, 1]), 0.5, 2000)
    theta = denormalized_theta(miles, price, theta)
    p = predict_(x, theta)
    print("loss :", loss_(p, y))
    save_model(theta)
    draw_regression(miles, price, theta)
