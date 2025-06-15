# 🏇 Máximo Número de Caballos en Ajedrez sin Ataques Mutuos

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Algorithm](https://img.shields.io/badge/Algorithm-A*%20%26%20Branch%20&%20Bound-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📜 Descripción
Implementación en Python de algoritmos de búsqueda (A* y Branch & Bound) para resolver el problema de colocar el máximo número de caballos en un tablero de ajedrez sin que se amenacen entre sí. Desarrollado como proyecto para la asignatura de Inteligencia Artificial.

## 🎯 Problema Matemático
Dado un tablero de M×N:
- **Objetivo**: Maximizar el número de caballos (k)
- **Restricción**: Ningún caballo puede atacar a otro
- **Movimiento del caballo**: L (2 casillas en una dirección + 1 en perpendicular)

## 🧠 Solución Implementada
### Algoritmos Clave
```python
# Búsqueda con Branch & Bound
search_horse_byb(initial_board)

# Búsqueda con A*
search_horse_astar(initial_board, heuristic_1|heuristic_2)
