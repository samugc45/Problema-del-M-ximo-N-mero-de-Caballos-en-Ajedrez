import numpy as np
#Crea un tablero vacío de tamaño MxN representado por una matriz de ceros, gracias a la librería de numpy, con la
#funcion .zeros inicializando todos sus elementos a 0 (el 0 representa posicion vacia y sin ser atacada)
def initial_state(M, N):
    return np.zeros((M, N), dtype=int)

# Ejemplo de uso de la función estado inicial
board = initial_state(3, 3)
print(board)

#Con una lista de tuplas, definimos los 8 posibles movimientos que tiene un caballo
#en un tablero de ajedrez
MOVIMIENTOS = [
    (2, 1), (2, -1),
    (-2, 1), (-2, -1),
    (1, 2), (1, -2),
    (-1, 2), (-1, -2)
]

#Con la función colocar_caballo, colocamos un caballo (1) en las posiciones i,j
#Además para esa posicón en la que se ha establecido el caballo, también colocamos las posiciones
#donde puede saltar el caballo como (-1), es decir, posición atacada por caballo,
#y no válida para colocar otro

def colocar_caballo(board, fila, columna):
    board[fila][columna] = 1
    for movimiento in MOVIMIENTOS:
        NuevaFila, NuevaColumna = fila + movimiento[0], columna + movimiento[1]
        if 0 <= NuevaFila < board.shape[0] and 0 <= NuevaColumna < board.shape[1] and board[NuevaFila, NuevaColumna] == 0:
            board[NuevaFila, NuevaColumna] = -1
    return board

#Nos va aportanto todos los posibles tableros resultantes de colocar un caballo en una posición específica.
def expand(board):
    boards = []
    #Con el método .shape de numpy, obtenemos las dimensionbes de la matriz
    for fila in range(board.shape[0]):
        for columna in range(board.shape[1]):
            if board[fila][columna] == 0: #Si la casilla no está amenazada y también si no hay un caballo
                tablero_Copia = board.copy() #Crea una copia del tablero inicial
                colocar_caballo(tablero_Copia, fila, columna) #Coloca caballo en la posicion en la que se encuentra, y actualiza las casillas amenazadas
                boards.append(tablero_Copia) #Añade el tablero, a la lista de tableros (boards)
    return boards

expand(board) # Debe devolver una lista de tableros

#Esta funcion verifica si hay casillas vacias, en caso de que no haya, retorna true, si hay alguna false
def is_solution(board):
  #Con el método -any de numpy, verificamos que con que uno de todos los elementos de borad sea (0), es decir que se cumpla la condición,
  #devolverá True, si no, devuelve false
  sol = np.any(board == 0)
  return not sol

#Esta función calcula de cada camino o estado, su coste durante la búsqueda
def cost(path):
    board = path[-1]
    cost = 0
    cost = np.sum(board == -1)
    return cost

#Esta función calcula la heurística de los estados
def heuristic_1(board):
    heuristic = 0
    heuristic += np.sum(board == 0)  # Suma todas las casillas disponibles para colocar caballo (0)
    return heuristic

# Esta función calcula la heurística contando las casillas amenzadas por el caballo en los siguientes movimientos
def heuristic_2(board):

    movimientos_disponibles = 0  # Inicializamos los movimientos disponibles
    for fila in range(board.shape[0]):
        for columna in range(board.shape[1]):
            if board[fila, columna] == 0: # Solo casillas libres (0)
                for movimientos in MOVIMIENTOS:
                    NuevaFila, NuevaColumna = fila + movimientos[0], columna + movimientos[1]
                    # Contamos la casilla si está dentro del tablero y no está amenazada
                    if (
                        0 <= NuevaFila < board.shape[0]
                        and 0 <= NuevaColumna < board.shape[1]
                        and board[NuevaFila, NuevaColumna] == 0
                    ):
                        movimientos_disponibles += 1
    return movimientos_disponibles

#Esta función realiza una poda de los diferentes caminos según su coste (siempre el menor, el mejor)

def prune(path_list):
    mejores_caminos = {} #Se crea un diccionario que almacenará el mejor coste para cada estado final.
    for camino in path_list:
        estado = tuple(map(tuple, camino[-1])) #Se extrae el último tablero del camino para usarlo como clave
        #map es una funcion que convierte los elementos contenidos en el segundo
        #parametro en elementos del tipo indicado en el primer parametro
        #tuple convierte en una tupla al elemento pasado como argumento
        coste_actual = cost(camino) #Coste total del camino
        if estado not in mejores_caminos or coste_actual < mejores_caminos[estado]:  # Si el estado no se encuntra en el diccionario o si el coste del camino actual es menor
            mejores_caminos[estado] = coste_actual #Aquí actualizamos
    caminos_podados = [camino for camino in path_list if tuple(map(tuple, camino[-1])) in mejores_caminos]#Lista de caminos podados con menor coste
    return caminos_podados

# *args y **kwargs son argumentos variables, si el argumento no es reconocido es almacenado en estas variables.
# Aquí se utilizan para ignorar argumentos innecesarios.

def order_astar(old_paths, new_paths, c, h, *args, **kwargs):
    """Ordena la lista de caminos para A* utilizando el coste y heurística."""
    caminos_juntos = old_paths + new_paths#Juntamos los caminos viejos y nuevos
    caminos_juntos.sort(key=lambda camino: c(camino) + h(camino[-1]))# Ordenamos según coste y heurística
    return prune(caminos_juntos)# Podamos caminos

def order_byb(old_paths, new_paths, c, *args, **kwargs):
    """Ordena la lista de caminos para Branch & Bound utilizando solo el coste."""
    caminos_juntos = old_paths + new_paths# Juntamos los caminos viejos y nuevos
    caminos_juntos.sort(key=lambda camino: c(camino))# Ordenamos según el coste únicamente
    return prune(caminos_juntos)# Podamos caminos

#Esta función se encarga de buscar la solución más ópmita teniendo en cuenta que los caballos no se amenacen entre sí
#En ella se utiliza un algoritmo de búsqueda de expansión de caminos, teniendo en cuenta tanto el coste como la heurística y seleccionando el menor (más óptimo)
def search(initial_board, expansion, cost, heuristic, ordering, solution):

    paths = [[initial_board]] # Se inicializa la lista de caminos con el estado inicial
    mejor_solucion = None # Variable que almacena la mejor solución
    mejor_num_caballos = 0 # Variable que contabiliza el numero de caballos en la mejor solución hallada hasta el momento
    visitados = set()  # Para evitar estados repetidos

    while paths: # Se ejecuta mientras haya caminos por explorar
        primer_elemento = paths.pop(0) # Primer camino
        estado_actual = tuple(map(tuple, primer_elemento[-1])) # El último tablero del camino se convierte a una tupla para evitar que se duplique

        '''
        map es una funcion que convierte los elementos contenidos en el segundo
        parametro en elementos del tipo indicado en el primer parametro

        tuple convierte en una tupla al elemento pasado como argumento
        '''

        # Ignora el estado si ya ha sido visitado
        if estado_actual in visitados:
            continue
        visitados.add(estado_actual)

        # Verifica si el estado actual es una solución
        if solution(primer_elemento[-1]):
            num_caballos = np.sum(primer_elemento[-1] == 1)
            if num_caballos > mejor_num_caballos:
                mejor_num_caballos = num_caballos
                mejor_solucion = primer_elemento[-1]
            continue

        # Expande el estado actual y ordena los caminos
        nuevos_estados = expansion(primer_elemento[-1])
        if nuevos_estados:
            nuevos_caminos = [primer_elemento + [estado] for estado in nuevos_estados]
            paths = ordering(paths, nuevos_caminos, cost, heuristic or (lambda x: 0))

    return mejor_solucion

################################# NO TOCAR #################################
#                                                                          #
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("Executime time: ", end - start, " seconds")
        return res
    return wrapper
#                                                                          #
################################# NO TOCAR #################################

# Este codigo temporiza la ejecución de una función cualquiera

################################# NO TOCAR #################################
#                                                                          #
@timer
def search_horse_byb(initial_board):
    return search(initial_board, expand, cost, None, order_byb, is_solution)

@timer
def search_horse_astar(initial_board, heuristic):
    return search(initial_board, expand, cost, heuristic, order_astar, is_solution)
#                                                                          #
################################# NO TOCAR #################################

CONF = {'2x2': (2, 2),
        '3x3': (3, 3),
        '3x5': (3, 5),
        '5x5': (5, 5),
        '8x8': (8, 8),
        }

def measure_solution(board):
    return np.sum(board == 1)

def launch_experiment(configuration, heuristic=None):
    conf = CONF[configuration]
    print(f"Running {'A*' if heuristic else 'B&B'} with {configuration} board")
    if heuristic:
        sol = search_horse_astar(initial_state(*conf), heuristic)
    else:
        sol = search_horse_byb(initial_state(*conf))
    n_c = measure_solution(sol)
    print(f"Solution found: \n{sol}")
    print(f"Number of horses in solution: {n_c}")

    return sol, n_c

launch_experiment('2x2') # Ejemplo de uso para B&B
print()
launch_experiment('2x2', heuristic=heuristic_2) # Ejemplo de uso para A*
print()
print("Execution finished")

### Coloca aquí tus experimentos ###
launch_experiment('2x2') # B&B 2X2
print()
launch_experiment('3x3') # B&B 3X3
print()
launch_experiment('3x5') # B&B 3X5
print()
launch_experiment('5x5') # B&B 5X5
print()
launch_experiment('8x8') # B&B 8X8