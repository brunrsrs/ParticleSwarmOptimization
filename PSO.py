#PSO
import numpy as np
import matplotlib.pyplot as plt

# função que busca minimzar
def funcao_objetiva(posicao):
    x, y = posicao
    return (1 - x)**2 + 100 * (y - x**2)**2

# ---------------------------------------------------

# implementação do PSO
def PSO(max_it, N, AC1, AC2, Vmax, Vmin, limites):
    X = np.random.uniform(limites[0], limites[1], (N, 2))
    v = np.random.uniform(Vmin, Vmax, (N, 2))

    p = X.copy()
    t = 0
    melhores = []
    media = []
    for t in range(max_it):
        for i in range(N):
            # calcula o menor valor local
            if funcao_objetiva(X[i]) < funcao_objetiva(p[i]):
                p[i] = X[i].copy()
            g = i
            # calcula o menor valor global
            for j in range(N):
                if funcao_objetiva(p[j]) < funcao_objetiva(p[g]):
                    g = j

            r1 = np.random.uniform(0, AC1, 2)
            r2 = np.random.uniform(0, AC2, 2)
            #novas velocidades
            v[i] = v[i] +  r1 * (p[i] - X[i]) +  r2 * (p[g] - X[i])
            v[i] = np.clip(v[i], Vmin, Vmax)
            X[i] = X[i] + v[i]
            X[i] = np.clip(X[i], limites[0], limites[1])
        soma = 0
        for k in range(N):
            soma += funcao_objetiva(X[k])
        media.append(soma/N)
        melhores.append(funcao_objetiva(p[g]))

    return X, p, g, melhores, media

# ---------------------------------------------------

max_it = 100
N = 50 # numero de particulas
AC1 = AC2 = 2.05 # valores aleatorios
Vmax = 2 # velocidade maxima
Vmin = -2 # velocidade minima
limites = (-5, 5) # limites do plano

X, p, g, melhores, media = PSO(max_it, N, AC1, AC2, Vmax, Vmin, limites)

print("Melhor posição encontrada:", p[g])
print("Melhor valor da função objetivo:", funcao_objetiva(p[g]))

# plotar o grafico
plt.plot(melhores)
plt.xlabel('Iterações')
plt.ylabel('Melhor valor da função objetivo')
plt.title('Convergência do PSO para melhores valores')
plt.show()

plt.plot(media)
plt.xlabel('Iterações')
plt.ylabel('Média dos valores da função objetivo')
plt.title('Média dos valores encontrados')
plt.show()