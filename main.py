import numpy as np
import matplotlib.pyplot as plt
import csv


# from mpl_toolkits.mplot3d import Axes3D


# Funkcja generująca dane treningowe
def AEproj3_data():
    x = np.array([
        [2.47644821504290, 0.987362839898902, -0.664849704411987],
        [1.59551703887844, 3.47373366726014, 0.0713212995662683],
        [5.09686686764456, 2.13565744692904, 0.556255334992036],
        [3.15085007454813, 2.71303284168429, -0.337282892820353],
        [2.33024370969781, 2.67484483132225, -2.07320713742233],
        [2.04868165400661, 3.34126158056676, -0.00463825234415512],
        [0.968748344022725, 2.12984305030536, -0.104913737297717],
        [1.99735489650914, 2.67666606950934, 1.27602321251281],
        [1.12065189958417, 2.51502883161021, 2.63541125502063],
        [1.91220776181711, 1.62784935458932, 1.24386459820817],
        [-2.47644821504290, -0.987362839898902, 0.664849704411987],
        [-1.59551703887844, -3.47373366726014, -0.0713212995662683],
        [-5.09686686764456, -2.13565744692904, -0.556255334992036],
        [-3.15085007454813, -2.71303284168429, 0.337282892820353],
        [-2.33024370969781, -2.67484483132225, 2.07320713742233],
        [-2.04868165400661, -3.34126158056676, 0.00463825234415512],
        [-0.968748344022725, -2.12984305030536, 0.104913737297717],
        [-1.99735489650914, -2.67666606950934, -1.27602321251281],
        [-1.12065189958417, -2.51502883161021, -2.63541125502063],
        [-1.91220776181711, -1.62784935458932, -1.24386459820817]
    ])
    y = np.array([
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ])
    return x, y


def plotowanie(Xx, Yy, ww, bb):
    # Wykres 3D z punktami treningowymi i płaszczyzną klasyfikatora
    tmp = np.linspace(-5, 5, 30)
    x, y = np.meshgrid(tmp, tmp)
    z = (-bb - ww[0] * x - ww[1] * y) / ww[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Punkty treningowe
    ax.scatter(Xx[:, 0], Xx[:, 1], Xx[:, 2], c=Yy)

    # Płaszczyzna klasyfikatora

    # Konfiguracja wykresu
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Linear Classifier')
    ax.plot_surface(x, y, z, alpha=0.5, cmap='viridis')
    # Wyświetlanie wykresu
    for angle in range(0, 360, 30):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(0.001)
    plt.show()


plt.close('all')
X, Y = AEproj3_data()
l_danych, l_cech = X.shape
w = np.zeros(l_cech)
b = 0
r = np.max(np.linalg.norm(X, axis=1))
eta = 0.001
w[2] = 0.000000001
plotowanie(X, Y, w, b)
w[2] = 0
errors = []
for _ in range(100):
    error = 0
    for i in range(l_danych):
        if np.sign(np.dot(w, X[i, :]) - b) != Y[i]:
            w = w + eta * Y[i] * X[i, :]
            b = b - eta * Y[i] * r * r
            error += 1
            plotowanie(X, Y, w, b)
            print(w)
            print(b)
        errors.append(error)
        if error == 0:
            break

print(errors)
plotowanie(X, Y, w, b)
plt.plot(errors)
plt.show()
