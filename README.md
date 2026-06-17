# Cuckoo Search Algorithm (CSA) con OpenMP

Implementación paralela del **Algoritmo de Búsqueda del Cucú** en C++ con OpenMP, aplicado a funciones de prueba clásicas y a la estimación de parámetros de una celda de combustible.

## Descripción

El Cuckoo Search es una metaheurística bio-inspirada en el comportamiento de parasitismo de nidos del cucú. Cada solución candidata ("nido") se actualiza mediante **vuelos de Lévy**, que combinan pasos locales pequeños con saltos largos ocasionales, lo que le da buenas propiedades de exploración.

## Funciones de prueba

| Función | Dominio | Mínimo conocido |
|---|---|---|
| Sinusoidal | [-2, 2]² | 0.1 |
| Michalewicz | [0, 4]² | ≈ -1.801 |
| Rosenbrock | [0, 4]² | 0.0 |
| Six-hump Camelback | [-3,3]×[-2,2] | ≈ -1.0316 |

## Aplicación: Estimación de parámetros de celda de combustible

Se calibran 5 parámetros del modelo de voltaje de una celda PEM minimizando tres métricas de error:

- **SSE** — Suma de cuadrados
- **SAE** — Suma de valores absolutos  
- **MAE** — Mediana de errores absolutos

## Compilación

```bash
# Con OpenMP (recomendado)
g++ -O3 -fopenmp -o cuckoo cuckoo_search.cpp

# Sin OpenMP
g++ -O3 -o cuckoo cuckoo_search.cpp
```

## Uso

```bash
./cuckoo [funcion] [hilos]

./cuckoo all 4           # Benchmarks + speedup + celda de combustible
./cuckoo benchmark 4     # Solo benchmarks
./cuckoo speedup 4       # Análisis de speed-up (1, 2, 4, 8 hilos)
./cuckoo fuelcell 4      # Estimación de parámetros
./cuckoo rosenbrock 4    # Una función específica
./cuckoo anim 4          # Genera datos para animación
```

## Análisis de speed-up

El programa incluye un modo de análisis de escalabilidad que mide el tiempo con 1, 2, 4 y 8 hilos y calcula speed-up y eficiencia:

```
Hilos     Tiempo(s)     Speed-up    Eficiencia
1         12.3420       1.000       100.0%
2          6.4210       1.922        96.1%
4          3.3150       3.723        93.1%
8          1.9840       6.222        77.8%
```

## Parámetros del algoritmo

| Parámetro | Descripción | Default |
|---|---|---|
| `pop_size` | Número de nidos | 4000 |
| `max_iter` | Iteraciones máximas | 500 |
| `pa` | Fracción de nidos abandonados | 0.25 |
| `alpha` | Escala del vuelo de Lévy | 0.01 |
| `beta` | Exponente de la distribución de Lévy | 1.5 |

## Tecnologías

- C++17
- OpenMP (paralelismo a nivel de población)
- Algoritmo de Mantegna para muestreo eficiente de vuelos de Lévy
