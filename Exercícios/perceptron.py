import numpy as np
import pandas as pd
import math

def predicao(w, x, bias):
    somatorio = np.dot(w, x.T) + bias
    return stepFunction(somatorio)

def stepFunction(somatorio):
    return 1 if somatorio >= 0 else 0

def perceptron(X, Y, pesos = [0, 0], bias = 0, alpha = 0.1, epocas = 5):
    convergiu = False
    while not convergiu:
        for index, x in enumerate(X):
            erro = Y[index][0] - predicao(pesos, x, bias)
            bias = bias + alpha * erro
            for index_p, p in enumerate(pesos):
                peso_novo = pesos[index_p] + (alpha*erro*x[index_p])
                #if math.isclose(p, peso_novo, rel_tol=1e-3): #verifica se convergiu
                if erro == 0:
                    convergiu = True
                else:
                    pesos[index_p] = peso_novo
                    convergiu = False
    return pesos, bias

entradas = [[2.7810836, 2.550537003, 0],
            [1.465489372, 2.362125076, 0],
            [3.396561688, 4.400293529, 0],
            [1.38807019, 1.850220317, 0],
            [3.06407232, 3.005305973, 0],
            [7.627531214, 2.759262235, 1],
            [5.332441248, 2.088626775, 1],
            [6.922596716, 1.77106367, 1],
            [8.675418651, -0.242068655, 1],
            [7.673756466, 3.508563011, 1]]

dados = pd.DataFrame(entradas, columns=['x0', 'x1', 'saida'])
X = np.array(dados[['x0','x1']].copy())
Y = np.array(dados[['saida']].copy())

epocas = 5
pesos =[0, 0]
bias = 0
for epoca in range(epocas):
    pesos, bias = perceptron(X, Y, pesos = pesos, bias = bias)
    print(pesos, bias)
