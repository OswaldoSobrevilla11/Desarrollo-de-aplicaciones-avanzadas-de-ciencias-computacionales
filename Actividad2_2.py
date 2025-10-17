# Actividad 2.2 realizada por Jose Oswaldo Sobrevilla Vazquez A01412742
# ================================================
# Red Hopfield general (10x10) sin librerías externas
# ================================================

# --- Definir figuras 10x10 (simples emojis o letras) ---
# 1 = activo, 0 = inactivo

figuras = [
    # 😀 Carita feliz
    [
        [0,0,1,1,1,1,1,1,0,0],
        [0,1,0,0,0,0,0,0,1,0],
        [1,0,1,0,0,0,0,1,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,1,0,0,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,1,0,0,0,0,1,0,1],
        [0,1,0,1,1,1,1,0,1,0],
        [0,0,1,0,0,0,0,1,0,0],
        [0,0,0,1,1,1,1,0,0,0]
    ],
    # ❤️ Corazón
    [
        [0,1,1,0,0,0,1,1,0,0],
        [1,1,1,1,0,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,0,0],
        [0,0,1,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]
    ],
    # 🔺 Triángulo
    [
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,0],
        [0,0,1,1,1,1,1,0,0,0],
        [0,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]
    ],
    # ➕ Cruz
    [
        [0,0,0,1,1,1,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [0,0,0,1,1,1,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]
    ],
    # ⬜ Cuadro
    [
        [1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,0]
    ],
    # 🌀 Espiral
    [
        [1,1,1,1,1,1,1,1,1,0],
        [1,0,0,0,0,0,0,0,1,0],
        [1,0,1,1,1,1,1,0,1,0],
        [1,0,1,0,0,0,1,0,1,0],
        [1,0,1,0,1,0,1,0,1,0],
        [1,0,1,0,0,0,1,0,1,0],
        [1,0,1,1,1,1,1,0,1,0],
        [1,0,0,0,0,0,0,0,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0]
    ]
]

# --- Conversión a vector +1/-1 ---
def bin_to_pm1(fig):
    vec = []
    for fila in fig:
        for v in fila:
            vec.append(1 if v == 1 else -1)
    return vec

patrones = [bin_to_pm1(f) for f in figuras]

# --- Crear matriz de pesos ---
N = len(patrones[0])
W = [[0 for _ in range(N)] for _ in range(N)]

for X in patrones:
    for i in range(N):
        for j in range(N):
            W[i][j] += X[i] * X[j]

# Eliminar auto-conexiones
for i in range(N):
    W[i][i] = 0

print(f"\nMatriz de pesos construida (N={N})")

# --- Función de recuperación ---
def hopfield_recall(entrada, W, max_iter=10):
    estado = entrada[:]
    N = len(estado)
    for _ in range(max_iter):
        for i in range(N):
            s = 0
            for j in range(N):
                s += W[i][j] * estado[j]
            estado[i] = 1 if s >= 0 else -1
    return estado

# --- Mostrar figura ---
def mostrar_figura(vector):
    for i in range(10):
        fila = ""
        for j in range(10):
            if vector[i*10 + j] == 1:
                fila += "█"
            else:
                fila += " "
        print(fila)
    print()

# --- Pruebas ---
print("\n=== PRUEBAS ===")

# 1. Figura original
entrada = patrones[0][:]
print("Figura original (1):")
mostrar_figura(entrada)
salida = hopfield_recall(entrada, W)
print("Figura recuperada:")
mostrar_figura(salida)

# 2. Figura con ruido
entrada_ruido = patrones[1][:]
entrada_ruido[12] *= -1
entrada_ruido[45] *= -1
entrada_ruido[78] *= -1
print("Figura 2 con ruido:")
mostrar_figura(entrada_ruido)
salida2 = hopfield_recall(entrada_ruido, W)
print("Figura recuperada:")
mostrar_figura(salida2)
