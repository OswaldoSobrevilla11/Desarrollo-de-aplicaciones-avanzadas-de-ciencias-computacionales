# Actividad 2.1 realizada por Jose Oswaldo Sobrevilla Vazquez A01412742
# Red Hopfield con listas y ciclos

# Definir patrones
X1 = [1, 1, 1, -1]
X2 = [-1, -1, -1, 1]

# Número de neuronas (longitud de los patrones)
n = len(X1)

# Inicializacion matriz de pesos con ceros
W = [[0 for _ in range(n)] for _ in range(n)]

# Regla de Hebb: W = X1^T * X1 + X2^T * X2
patrones = [X1, X2]

for X in patrones:
    for i in range(n):
        for j in range(n):
            W[i][j] += X[i] * X[j]

# Eliminar auto-conexiones (diagonal en cero)
for i in range(n):
    W[i][i] = 0

# Mostrar matriz de pesos
print("Matriz de pesos W:")
for fila in W:
    print(fila)


# Función de recuperación
def hopfield_recall(entrada, W, max_iter=10):
    estado = entrada[:]
    for _ in range(max_iter):
        for i in range(len(estado)):
            # Calcular suma ponderada
            suma = 0
            for j in range(len(estado)):
                suma += W[i][j] * estado[j]

            # Función signo
            if suma >= 0:
                estado[i] = 1
            else:
                estado[i] = -1
    return estado


# --- Pruebas ---
print("\nRecuperando X1:")
print(hopfield_recall([1, 1, 1, -1], W))

print("\nRecuperando X2:")
print(hopfield_recall([-1, -1, -1, 1], W))

print("\nProbando con un patrón ruidoso:")
print(hopfield_recall([-1, -1, -1, -1], W))
