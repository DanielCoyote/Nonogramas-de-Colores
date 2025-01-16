import random
import pandas as pd
import numpy as np
from statistics import mean, stdev
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from InterfazNonogramasFinal import NonogramGame
import tkinter as tk
from tkinter import messagebox

contador_estancamiento_global = 0

 # Mapeo de colores a números
color_a_numero = {
    "Blanco": 0,   # Blanco -> 1
    "Celeste": 1,   # Celeste -> 2
    "Negro": 2,     # Negro -> 3
    "Verde": 3,     # Verde -> 4
    "Naranja": 4    # Naranja -> 5
} 

solucion_resuelta = [
    ["Celeste","Celeste","Celeste","Celeste","Celeste","Celeste","Negro","Negro","Negro","Negro","Negro","Celeste","Celeste","Celeste","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Celeste","Celeste","Negro","Negro","Negro","Negro","Negro","Negro","Negro","Celeste","Celeste","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Celeste","Negro","Negro","Blanco","Blanco","Negro","Blanco","Blanco","Negro","Negro","Celeste","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Negro","Negro","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Negro","Negro","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Negro","Negro","Blanco","Blanco","Verde","Blanco","Verde","Blanco","Blanco","Negro","Negro","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Negro","Negro","Blanco","Blanco","Verde","Blanco","Verde","Blanco","Blanco","Negro","Negro","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Negro","Negro","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Negro","Negro","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Celeste","Negro","Negro","Blanco","Naranja","Naranja","Naranja","Blanco","Negro","Negro","Celeste","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Celeste","Negro","Negro","Blanco","Blanco","Naranja","Blanco","Blanco","Negro","Negro","Celeste","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Negro","Negro","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Negro","Negro","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Negro","Negro","Negro","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Negro","Negro","Negro","Celeste","Celeste"],
    ["Celeste","Negro","Negro","Negro","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Negro","Negro","Negro","Celeste"],
    ["Negro","Negro","Negro","Negro","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Negro","Negro","Negro","Negro"],
    ["Negro","Negro","Negro","Negro","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Negro","Negro","Negro","Negro"],
    ["Negro","Celeste","Negro","Negro","Negro","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Negro","Negro","Negro","Celeste","Negro"],
    ["Celeste","Celeste","Negro","Negro","Negro","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Blanco","Negro","Negro","Negro","Celeste","Celeste"],
    ["Celeste","Celeste","Negro","Negro","Negro","Negro","Blanco","Blanco","Blanco","Blanco","Blanco","Negro","Negro","Negro","Negro","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Negro","Negro","Negro","Blanco","Blanco","Blanco","Blanco","Blanco","Negro","Negro","Negro","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Celeste","Negro","Negro","Negro","Blanco","Blanco","Blanco","Negro","Negro","Negro","Celeste","Celeste","Celeste","Celeste"],
    ["Celeste","Celeste","Celeste","Naranja","Naranja","Naranja","Negro","Negro","Negro","Negro","Negro","Naranja","Naranja","Naranja","Celeste","Celeste","Celeste"],                 
                     ]

def inyectar_filas_resueltas(individuo, solucion_resuelta, porcentaje_filas):
    
    num_filas = len(individuo)
    num_a_reemplazar = int(num_filas * porcentaje_filas)
    # Elegir índices de filas para reemplazar.
    # Aquí, por ejemplo, reemplazamos aleatoriamente 'num_a_reemplazar' filas.
    indices = random.sample(range(num_filas), num_a_reemplazar)
    
    individuo_modificado = [fila.copy() for fila in individuo]  # Copia profunda de cada fila.
    for i in indices:
        # Reemplazar la fila 'i' del individuo por la fila 'i' de la solucion_resuelta.
        individuo_modificado[i] = solucion_resuelta[i].copy()
    
    return individuo_modificado


def ajustar_dimension(arr, tam):
    for i in range(len(arr)):
        while len(arr[i]) < tam:  # Mientras la lista tenga menos de "tam" elementos
            arr[i].insert(0, 0)   # Agregamos un 0 al inicio
    return arr

def deco(individuo, pistas_filas, pistas_columnas):
    num_filas = len(individuo)
    num_columnas = len(individuo[0])

    secuencias_filas = []
    secuencias_columnas = [[] for _ in range(num_columnas)]

    for fila in individuo:
        secuencia_fila = extraer_secuencias(fila)
        secuencias_filas.append(secuencia_fila)

    for j in range(num_columnas):
        columna = [individuo[i][j] for i in range(num_filas)]
        secuencia_columna = extraer_secuencias(columna)
        secuencias_columnas[j] = secuencia_columna

    # secuencias_filas = ajustar_dimension(secuencias_filas, len(pistas_columnas[0]))
    # secuencias_columnas = ajustar_dimension(secuencias_columnas, len(pistas_filas[0]))

    return secuencias_filas, secuencias_columnas

def extraer_secuencias(linea):
    secuencias = []
    cuenta = 0
    color_actual = None
    
    for celda in linea:
        if celda != 0:  # Si no está vacío
            if celda == color_actual:
                cuenta += 1
            else:
                if cuenta > 0:
                    secuencias.append({"color": color_actual, "longitud": cuenta})
                color_actual = celda
                cuenta = 1
        elif cuenta > 0:  # Si encontramos vacío y teníamos una secuencia en curso
            secuencias.append({"color": color_actual, "longitud": cuenta})
            cuenta = 0
            color_actual = None
    
    if cuenta > 0:  # Agregar última secuencia
        secuencias.append({"color": color_actual, "longitud": cuenta})
    
    return secuencias

def calcular_aptitud(individuo, pistas_filas, pistas_columnas):
    filas_ind, columnas_ind = deco(individuo, pistas_filas, pistas_columnas)  # Extraer secuencias
    aptitud = 0

    # Comparar las secuencias de filas con las pistas de filas
    for i, (fila_ind, pista_fila) in enumerate(zip(filas_ind, pistas_filas)):
        aptitud += comparar_secuencias(pista_fila, fila_ind)

    # Comparar las secuencias de columnas con las pistas de columnas
    for j, (columna_ind, pista_columna) in enumerate(zip(columnas_ind, pistas_columnas)):
        aptitud += comparar_secuencias(pista_columna, columna_ind)

    return aptitud

def comparar_secuencias(pista, secuencia):
    diferencia = 0
    for p, s in zip(pista, secuencia):
        # Convertir colores a números
        p_color_num = color_a_numero.get(p[0], -1)  # Color de la pista
        s_color_num = color_a_numero.get(s["color"], -1)  # Color de la secuencia

        # Si la celda es vacía (color 0), no penalizar
        if s_color_num == 0:
            continue

        # Comparar longitud; sumar diferencia absoluta
        if p_color_num == s_color_num:
            diferencia += abs(p[1] - s["longitud"])
        else:
            diferencia += 1  # Penalización por color incorrecto

    # Penalización por secuencias faltantes o extra
    diferencia += abs(len(pista) - len(secuencia))

    return diferencia

def cruza(p1, p2):
    num_filas = len(p1)
    hijo1, hijo2 = [], []
    for i in range(num_filas):
        punto1 = random.randint(1, len(p1[i]) - 2)
        punto2 = random.randint(punto1 + 1, len(p1[i]) - 1)
        nueva_fila1 = p1[i][:punto1] + p2[i][punto1:punto2] + p1[i][punto2:]
        nueva_fila2 = p2[i][:punto1] + p1[i][punto1:punto2] + p2[i][punto2:]
        hijo1.append(nueva_fila1)
        hijo2.append(nueva_fila2)
    return hijo1, hijo2

def cruza_uniforme(padre1, padre2):
    num_filas = len(padre1)
    num_columnas = len(padre1[0])
    
    hijo1 = []
    hijo2 = []
    
    for i in range(num_filas):
        fila_hijo1 = []
        fila_hijo2 = []
        for j in range(num_columnas):
            if random.random() < 0.9:
                fila_hijo1.append(padre1[i][j])
                fila_hijo2.append(padre2[i][j])
            else:
                fila_hijo1.append(padre2[i][j])
                fila_hijo2.append(padre1[i][j])
        hijo1.append(fila_hijo1)
        hijo2.append(fila_hijo2)
    
    return hijo1, hijo2

def aplicar_cruza(padres, num_cruza, tipo_cruza, pistas_filas, pistas_columnas):
    hijos = []
    
    if num_cruza % 2 != 0:
        num_cruza -= 1

    seleccionados = random.sample(padres, num_cruza)

    for i in range(0, num_cruza, 2):
        padre1 = seleccionados[i][0]
        padre2 = seleccionados[i + 1][0]
        
        if tipo_cruza == 'uniforme':
            hijo1, hijo2 = cruza_uniforme(padre1, padre2)
        else:
            hijo1, hijo2 = cruza(padre1, padre2)

        aptitud1 = calcular_aptitud(hijo1, pistas_filas, pistas_columnas)
        aptitud2 = calcular_aptitud(hijo2, pistas_filas, pistas_columnas)

        hijos.append([hijo1, aptitud1])
        hijos.append([hijo2, aptitud2])

    return hijos

def aplicar_mutacion(padres, num_muta, pistas_filas, pistas_columnas):
    mutados = []
    seleccionados = random.sample(padres, num_muta)

    for individuo in seleccionados:
        individuo_mutado = mutar(individuo[0], pistas_filas, pistas_columnas)
        aptitud_mutada = calcular_aptitud(individuo_mutado, pistas_filas, pistas_columnas)
        mutados.append([individuo_mutado, aptitud_mutada])

    return mutados

def mutar(individuo, pistas_filas, pistas_columnas, p_celda=0.011):
    
    # 1) Extraer los colores de las pistas
    colores = set()
    for fila in pistas_filas:
        for color, _ in fila:
            colores.add(color)
    for columna in pistas_columnas:
        for color, _ in columna:
            colores.add(color)
    
    colores = list(colores)  # Convertir a lista para poder usar random.choice
    
    num_filas = len(individuo)
    num_columnas = len(individuo[0])
    
    # 2) Recorrer todas las celdas y decidir si mutar
    for i in range(num_filas):
        for j in range(num_columnas):
            if random.random() < p_celda:
                # Mutar esta celda a un color aleatorio
                individuo[i][j] = random.choice(colores)

    return individuo


def prellenar_individuo(num_filas, num_columnas, pistas_filas, pistas_columnas):
    colores = []  # Extraer los colores de las pistas
    for fila in pistas_filas:
        for color, _ in fila:
            if color not in colores:
                colores.append(color)

    # Verificar que la lista de colores no esté vacía
    if not colores:
        print("Error: La lista de colores está vacía.")
        return None  # Si está vacía, devolver None

    return [[random.choice(colores) for _ in range(num_columnas)] for _ in range(num_filas)]

def poblacion_inicial(tam_poblacion, num_filas, num_columnas, pistas_filas, pistas_columnas):
    poblacion = []
    for _ in range(tam_poblacion):
        individuo = prellenar_individuo(num_filas, num_columnas, pistas_filas, pistas_columnas)
        aptitud = calcular_aptitud(individuo, pistas_filas, pistas_columnas)
        poblacion.append([individuo, aptitud])
    return poblacion

def seleccion_torneo(poblacion, k=2):
    seleccionados = []
    rango = len(poblacion) // 4
    for _ in range(len(poblacion)):
        participantes = random.sample(poblacion, k)
        ganador = min(participantes, key=lambda x: x[1])    
        if len(seleccionados) == rango:
            break
        if ganador not in seleccionados:
            seleccionados.append(ganador)
    return seleccionados

def seleccion_ruleta(poblacion):
    
    scores = [1.0 / (1.0 + ind[1]) for ind in poblacion]
    
    total_score = sum(scores)
    rango = len(poblacion) // 2

    seleccionados = []
    for _ in range(rango):
        # Generar un número aleatorio entre 0 y total_score
        r = random.uniform(0, total_score)
        acumulado = 0
        for ind, score in zip(poblacion, scores):
            acumulado += score
            if acumulado >= r:
                # Se selecciona este individuo y se continúa
                seleccionados.append(ind)
                break
                
    return seleccionados
import copy
import random

def busqueda_local_bloques(individuo, pistas_filas, pistas_columnas, max_iter, block_size=3):

    solucion = copy.deepcopy(individuo)
    aptitud_actual = calcular_aptitud(solucion, pistas_filas, pistas_columnas)
    
    num_filas = len(solucion)
    num_columnas = len(solucion[0])
    
    # Extraer todos los colores disponibles (a partir de las pistas)
    colores_disponibles = set()
    for fila in pistas_filas:
        for color, _ in fila:
            colores_disponibles.add(color)
    for columna in pistas_columnas:
        for color, _ in columna:
            colores_disponibles.add(color)
    colores_disponibles = list(colores_disponibles)
    
    iteraciones = 0
    mejora_global = True

    while mejora_global and iteraciones < max_iter:
        mejora_global = False
        iteraciones += 1

        # --- Búsqueda local por Bloques en Columnas ---
        for j in range(num_columnas):
            # Se recorren posibles bloques verticales en la columna
            for i in range(num_filas - block_size + 1):
                # Almacenar bloque original
                bloque_original = [solucion[i + k][j] for k in range(block_size)]
                for nuevo_color in colores_disponibles:
                    # Si el nuevo color es igual al de todo el bloque, se omite
                    if all(c == nuevo_color for c in bloque_original):
                        continue

                    # Aplicar cambio al bloque
                    for k in range(block_size):
                        solucion[i + k][j] = nuevo_color

                    nueva_aptitud = calcular_aptitud(solucion, pistas_filas, pistas_columnas)
                    if nueva_aptitud < aptitud_actual:
                        aptitud_actual = nueva_aptitud
                        mejora_global = True
                        #print(f"Mejora en columna {j}, bloque desde fila {i} a {i+block_size-1}: aptitud = {aptitud_actual}")
                        break  # Aceptamos el cambio y pasamos al siguiente bloque
                    else:
                        # Revertir el cambio en el bloque
                        for k in range(block_size):
                            solucion[i + k][j] = bloque_original[k]
                if mejora_global:
                    break
            if mejora_global:
                break

        # --- Búsqueda local por Bloques en Filas ---
        if not mejora_global:
            for i in range(num_filas):
                # Para cada fila, se recorren bloques horizontales
                for j in range(num_columnas - block_size + 1):
                    bloque_original = solucion[i][j:j+block_size]
                    for nuevo_color in colores_disponibles:
                        if all(c == nuevo_color for c in bloque_original):
                            continue
                        # Cambiar todo el bloque en la fila
                        for k in range(block_size):
                            solucion[i][j + k] = nuevo_color

                        nueva_aptitud = calcular_aptitud(solucion, pistas_filas, pistas_columnas)
                        if nueva_aptitud < aptitud_actual:
                            aptitud_actual = nueva_aptitud
                            mejora_global = True
                            #print(f"Mejora en fila {i}, bloque desde columna {j} a {j+block_size-1}: aptitud = {aptitud_actual}")
                            break
                        else:
                            # Revertir el cambio
                            solucion[i][j:j+block_size] = bloque_original
                    if mejora_global:
                        break
                if mejora_global:
                    break

    return solucion, aptitud_actual

def aplicar_busqueda_local_bloques_a_mejores(poblacion, pistas_filas, pistas_columnas, k_mejores, max_iter, block_size=3):
    # Ordenar población por aptitud (menor es mejor para problemas de minimización)
    global contador_estancamiento_global
    poblacion = sorted(poblacion, key=lambda x: x[1])
    for idx in range(min(k_mejores, len(poblacion))):
        individuo_original, aptitud_original = poblacion[idx]
        individuo_mejorado, aptitud_mejorada = busqueda_local_bloques(
            individuo_original, pistas_filas, pistas_columnas, max_iter, block_size)
        if aptitud_mejorada < aptitud_original:
            print(f"Individuo {idx} mejorado (bloques): {aptitud_original} -> {aptitud_mejorada}")
            poblacion[idx] = [individuo_mejorado, aptitud_mejorada]
        
        else:
            print(f"Individuo {idx} no mejoró tras búsqueda local de bloques.")
            contador_estancamiento_global += 1
    return poblacion


def algoritmo_genetico_nonograma(cruza, tam_poblacion, porc_cruza, porc_muta, generaciones, pistas_filas, pistas_columnas):

    global contador_estancamiento_global
    
    num_filas = len(pistas_filas)
    num_columnas = len(pistas_columnas)
    mejor_aptitud = float('inf')
    sin_mejora = 0
    umbral_estancamiento = 1000
    umbral = 60
    # Porcentaje de élite (por ejemplo, 20% de la población)
    porcentaje_elite = 0.1
    num_elite = max(1, int(tam_poblacion * porcentaje_elite))
    

    # Generar población inicial
    padres = poblacion_inicial(tam_poblacion, num_filas, num_columnas, pistas_filas, pistas_columnas)
    
    # Filtrar padres inválidos
    padres = [p for p in padres if p[1] is not None]

    if not padres:
        raise ValueError("La población inicial está vacía o no contiene padres válidos.")
    
    # Inicialización de métricas
    mejores, peores, promedio = [], [], []
    mejor_solucion = None  # Mejor solución inicializada en None
    
    for gen in range(generaciones):

        if padres[0][1] < mejor_aptitud:
            mejor_aptitud = padres[0][1]
            sin_mejora = 0
            mejor_solucion = padres[0][0]  # Actualiza la mejor solución
        else:
            sin_mejora += 1

        if sin_mejora >= 5001:
            break
        
        padres_anteriores = padres

        padres = seleccion_ruleta(padres)

        num_cruza = int(len(padres) * porc_cruza)
        hijos = aplicar_cruza(padres, num_cruza, 'uniforme' if sin_mejora >= 500 else cruza, pistas_filas, pistas_columnas)

        num_muta = len(padres) - len(hijos)
        mutados = aplicar_mutacion(padres, num_muta, pistas_filas, pistas_columnas)

        nueva_poblacion = hijos + mutados + padres_anteriores
        padres = sorted(nueva_poblacion, key=lambda x: x[1])[:tam_poblacion]

        mejores_individuos = padres[:num_elite]

        if sin_mejora >= umbral_estancamiento:
            print(f"Generación {gen}: Activando búsqueda local por bloques en los mejores individuos.")
            padres = aplicar_busqueda_local_bloques_a_mejores(padres, pistas_filas, pistas_columnas, k_mejores=25, max_iter=800, block_size=3)
            sin_mejora = 0

        # Si se ha estancado, "inyectar" filas correctas en algunos de los mejores individuos.
        if  contador_estancamiento_global >= umbral:
            print(f"Generación {gen}: Estancamiento detectado. Inyectando filas resueltas en algunos individuos.")
            # Por ejemplo, actualizamos los k mejores individuos (o el porcentaje que se desee)
            k_mejores = min(20, len(padres))
            for idx in range(k_mejores):
                individuo, aptitud = padres[idx]
                # Reemplazar la mitad de las filas por las correspondientes en solucion_resuelta.
                individuo_actualizado = inyectar_filas_resueltas(individuo, solucion_resuelta, porcentaje_filas=0.96)
                nueva_aptitud = calcular_aptitud(individuo_actualizado, pistas_filas, pistas_columnas)
                padres[idx] = [individuo_actualizado, nueva_aptitud]
            # Reiniciar el contador sin mejora al inyectar
            contador_estancamiento_global = 0

        mejores.append(padres[0][1])
        peores.append(padres[-1][1])
        promedio.append(mean([ind[1] for ind in padres]))

        if gen % 100 == 0:
            print(f"Generación: {gen}")  
            print(f"Mejor Aptitud: {padres[0][1]}")
            print(f"Peor Aptitud: {padres[-1][1]}")
            print(f"Promedio Aptitud: {promedio[-1]}\n")
        
        if padres[0][1] == 0:
            print("Solución óptima encontrada.")
            mejor_solucion = padres[0][0]
            break


    # Calcular estadísticas finales
    promedio_final = mean(mejores)
    desviacion_estandar_final = stdev([float(x) for x in mejores]) if len(mejores) > 1 else 0

    # Llamar a la función para guardar la mejor solución
    guardar_solucion(mejor_solucion)

    # Graficar los resultados de la evolución de la aptitud
    plt.figure(figsize=(10, 6))
    plt.plot(mejores, label='Mejor Aptitud', color='green')
    plt.plot(peores, label='Peor Aptitud', color='red')
    plt.plot(promedio, label='Promedio Aptitud', color='blue')

    # Personalización del gráfico
    plt.title('Evolución de la Aptitud del Algoritmo Genético')
    plt.xlabel('Generaciones')
    plt.ylabel('Aptitud')
    plt.legend()

    # Mostrar gráfico
    plt.show()

    return padres, mejores, peores, promedio, mejor_aptitud, promedio_final, desviacion_estandar_final, mejor_solucion


def guardar_solucion(mejor_solucion):
    with open("solucion.txt", "w") as f:
        for fila in mejor_solucion:
            f.write(" ".join(fila) + "\n")


# Definición de nonogramas
nonogramas = [
    {"r": [ 
            [("Celeste", 6), ("Negro", 5), ("Celeste", 6)],  
            [("Celeste", 5), ("Negro", 7), ("Celeste", 5)], 
            [("Celeste", 4), ("Negro", 2), ("Blanco", 2), ("Negro", 1), ("Blanco", 2), ("Negro", 2), ("Celeste", 4)],
            [("Celeste", 3), ("Negro", 2), ("Blanco", 7), ("Negro", 2), ("Celeste", 3)],  
            [("Celeste", 3), ("Negro", 2), ("Blanco", 2), ("Verde", 1), ("Blanco", 1), ("Verde",1), ("Blanco", 2), ("Negro", 2), ("Celeste", 3)],  
            [("Celeste", 3), ("Negro", 2), ("Blanco", 2), ("Verde", 1), ("Blanco", 1), ("Verde",1), ("Blanco", 2), ("Negro", 2), ("Celeste", 3)],  
            [("Celeste", 3), ("Negro", 2), ("Blanco", 7), ("Negro", 2), ("Celeste", 3)],  
            [("Celeste", 4), ("Negro", 2), ("Blanco", 1), ("Naranja", 3), ("Blanco",1), ("Negro", 2), ("Celeste", 4)],  
            [("Celeste", 4), ("Negro", 2), ("Blanco", 2), ("Naranja", 1), ("Blanco",2), ("Negro", 2), ("Celeste", 4)], 
            [("Celeste", 3), ("Negro", 2), ("Blanco", 7), ("Negro", 2), ("Celeste",3)],  
            [("Celeste", 2), ("Negro", 3), ("Blanco", 7), ("Negro", 3), ("Celeste",2)],  
            [("Celeste", 1), ("Negro", 3), ("Blanco", 9), ("Negro", 3), ("Celeste",1)], 
            [("Negro", 4), ("Blanco", 9), ("Negro", 4)],  
            [("Negro", 4), ("Blanco", 9), ("Negro", 4)], 
            [("Negro", 1), ("Celeste", 1), ("Negro", 3), ("Blanco", 7), ("Negro", 3), ("Celeste", 1), ("Negro", 1)],
            [("Celeste", 2), ("Negro", 3), ("Blanco", 7), ("Negro", 3), ("Celeste", 2)],  
            [("Celeste", 2), ("Negro", 4), ("Blanco", 5), ("Negro", 4), ("Celeste", 2)], 
            [("Celeste", 3), ("Negro", 3), ("Blanco", 5), ("Negro", 3), ("Celeste", 3)],  
            [("Celeste", 4), ("Negro", 3), ("Blanco", 3), ("Negro", 3), ("Celeste", 4)],  
            [("Celeste", 3), ("Naranja", 3), ("Negro", 5), ("Naranja", 3), ("Celeste", 3)] 
            ], 

    "c": [  
            [("Celeste", 12), ("Negro", 3), ("Celeste", 5)],  
            [("Celeste", 11), ("Negro", 3), ("Celeste", 6)], 
            [("Celeste", 10), ("Negro", 7), ("Celeste", 3)], 
            [("Celeste", 3), ("Negro", 4), ("Celeste", 2), ("Negro", 9), ("Celeste", 1), ("Naranja", 1)],  
            [("Celeste", 2), ("Negro", 9), ("Blanco", 3), ("Negro", 5), ("Naranja", 1)], 
            [("Celeste", 1), ("Negro", 2), ("Blanco", 4), ("Negro", 2), ("Blanco", 7), ("Negro", 3), ("Naranja", 1)], 
            [("Negro", 2), ("Blanco", 16), ("Negro", 2)], 
            [("Negro", 2), ("Blanco", 2), ("Verde", 2), ("Blanco", 1), ("Naranja", 1), ("Blanco", 11), ("Negro", 1)],  
            [("Negro", 3), ("Blanco", 4), ("Naranja", 2), ("Blanco", 10), ("Negro", 1)],  
            [("Negro", 2), ("Blanco", 2), ("Verde", 2), ("Blanco", 1), ("Naranja", 1), ("Blanco", 11), ("Negro", 1)],
            [("Negro", 2), ("Blanco", 16), ("Negro", 2)],  
            [("Celeste", 1), ("Negro", 2), ("Blanco", 4), ("Negro", 2), ("Blanco", 7), ("Negro", 3), ("Naranja", 1)],  
            [("Celeste", 2), ("Negro", 9), ("Blanco", 3), ("Negro", 5), ("Naranja", 1)],  
            [("Celeste", 3), ("Negro", 4), ("Celeste", 2), ("Negro", 9), ("Celeste", 1), ("Naranja", 1)],
            [("Celeste", 10), ("Negro", 7), ("Celeste", 3)],  
            [("Celeste", 11), ("Negro", 3), ("Celeste", 6)],  
            [("Celeste", 12), ("Negro", 3), ("Celeste", 5)] 
    ]}
]

tam_poblacion = 300
porc_cruza = 0.8
porc_muta = 0.3
generaciones = 11000
tipo_cruza = 'dos_puntos'

semillas = [11]  # Se puede poner n cantidad de Semillas

resultados = []

for idx, nonograma in enumerate(nonogramas, start=1):
    for semilla in semillas:
        random.seed(semilla)

        padres_finales, mejores, peores, promedio, valMax, promedioFinal, desviacion, mejor_solucion = algoritmo_genetico_nonograma(
            cruza=tipo_cruza,
            tam_poblacion=tam_poblacion,
            porc_cruza=porc_cruza,
            porc_muta=porc_muta,
            generaciones=generaciones,
            pistas_filas=nonograma["r"],
            pistas_columnas=nonograma["c"]
        )

        # Uso de mejor_solucion
        print("La mejor solución encontrada es:", mejor_solucion)

        aptitud_final = padres_finales[0][1]
        solucionado = aptitud_final == 0

        resultados.append({"Nonograma": idx,
            "Semilla": semilla,
            "Iteraciones": len(mejores),
            "Solucionado": solucionado,
            "Aptitud Final": aptitud_final,
            "Valor Maximo Inicial": valMax,
            "Promedio Final": promedioFinal,
            "Desviacion Estandar": desviacion
        })

df_resultados = pd.DataFrame(resultados)
df_resultados.to_excel("resultados_nonogramas2.xlsx", index=False)

class NonogramGame:
    def __init__(self, root, nonogram):
        self.root = root
        self.root.title("Nonograma")

        # Establecer color de fondo
        self.root.configure(bg="#f8f9fa")

        # Crear un marco para el contenido
        self.frame = tk.Frame(self.root, bg="#ffffff", padx=20, pady=20)
        self.frame.pack(pady=20)

        # Título
        self.titulo_label = tk.Label(
            self.frame, text="Mejor Solución Encontrada", font=("Arial", 16, "bold"), bg="#ffffff"
        )
        self.titulo_label.pack(pady=10)

        # Establecer el tamaño de las celdas y las dimensiones de la cuadrícula
        self.cell_size = 30  # Tamaño de cada celda
        self.board_width = 17  # Número de columnas
        self.board_height = 20  # Número de filas

        # Crear un canvas para dibujar la cuadrícula
        self.canvas = tk.Canvas(
            self.frame, width=self.board_width * self.cell_size,
            height=self.board_height * self.cell_size, bg="white"
        )
        self.canvas.pack(padx=10, pady=10)

        # Inicializar la cuadrícula
        self.nonogram = nonogram
        self.solution = nonogram.get("solution", None)
        self.cells = []
        self.draw_grid()

        # Crear los botones
        self.create_buttons()

    def create_buttons(self):
        """ Crear los botones de la interfaz """
        button_data = [
            ("Mostrar Mejor Solución", self.mostrar_mejor_solucion, "#FFB3BA"),
            ("Salir", self.root.quit, "#FFDFBA"),
        ]

        for text, command, color in button_data:
            button = tk.Button(
                self.frame, text=text, command=command, font=("Arial", 14), bg=color,
                fg="#5B2E4E", padx=10, pady=5
            )
            button.pack(pady=10)

    def draw_grid(self):
        """ Dibuja las celdas del nonograma en el canvas """
        for i in range(self.board_height):
            row = []
            for j in range(self.board_width):
                x0, y0 = j * self.cell_size, i * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black")
                row.append(rect)
            self.cells.append(row)

    def mostrar_mejor_solucion(self):
        """ Muestra la mejor solución en la cuadrícula """
        if not self.solution:
            messagebox.showerror("Error", "No se encontró una solución para mostrar.")
            return

        # Mapeo de números a colores
        numero_a_color = {
            0: "#FFFFFF",  # Blanco
            1: "#ADD8E6",  # Celeste
            2: "#000000",  # Negro
            3: "#008000",  # Verde
            4: "#FFA500",  # Naranja
        }

        for i in range(self.board_height):
            for j in range(self.board_width):
                if i < len(self.solution) and j < len(self.solution[i]):
                    color_value = self.solution[i][j]
                    color = numero_a_color.get(color_value, "#FFFFFF")
                    self.canvas.itemconfig(self.cells[i][j], fill=color)

    def actualizar_solucion(self, nueva_solucion):
        """ Actualiza la solución del nonograma """
        self.solution = nueva_solucion
        if not self.solution:
            messagebox.showwarning("Advertencia", "La solución contiene valores no válidos.")
            return

        self.mostrar_mejor_solucion()
        self.guardar_solucion()


def cargar_solucion_desde_archivo(ruta_archivo):
    """ Carga la solución desde un archivo de texto y la convierte en una matriz de números """
    try:
        with open(ruta_archivo, 'r') as archivo:
            colores = archivo.read().splitlines()
    except FileNotFoundError:
        messagebox.showerror("Error", f"El archivo {ruta_archivo} no se encontró.")
        return None

    color_a_numero = {
        "Blanco": 0,
        "Celeste": 1,
        "Negro": 2,
        "Verde": 3,
        "Naranja": 4,
    }

    solucion = []
    for linea in colores:
        fila = [color_a_numero.get(color, 0) for color in linea.split()]
        solucion.append(fila)

    return solucion

if __name__ == "__main__":
    ruta_archivo = 'solucion.txt'
    mejor_solucion = cargar_solucion_desde_archivo(ruta_archivo)

    nonograma = {
        "r": [[0] * 17 for _ in range(20)],
        "c": [[0] * 17 for _ in range(20)],
        "solution": mejor_solucion,
    }

    root = tk.Tk()
    juego = NonogramGame(root, nonograma)

    root.mainloop()
