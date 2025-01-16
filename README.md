# Nonogramas-de-Colores
Este repositorio contiene un solucionador de nonogramas en color que utiliza algoritmos genéticos junto con técnicas híbridas de búsqueda local (por bloques y por filas/columnas). 

## Características

- **Algoritmo Genético:**  
  Utiliza operadores de selección, cruza (incluyendo cruza uniforme y de dos puntos) y mutación para explorar el espacio de soluciones.

- **Búsqueda Local por Bloques y Filas/Columnas:**  
  Se emplean técnicas de refinamiento que modifican bloques (o regiones) de celdas para introducir cambios más significativos, ayudando a escapar de óptimos locales.

- **Manejo de Espacios Vacíos:**  
  Se extraen y comparan secuencias (bloques de celdas coloreadas) junto con la posición de inicio y fin, lo cual permite penalizar aquellas soluciones donde no se respeta la separación mínima (espacios vacíos) requerida por las pistas.

- **Interfaz Gráfica:**  
  Incluye una GUI desarrollada con Tkinter para visualizar y analizar la mejor solución encontrada.

## Requisitos

- Python 3.x
- Bibliotecas: `random`, `pandas`, `numpy`, `statistics`, `matplotlib`, `tkinter`

## Uso

1. Clona el repositorio:

   ```bash
   git clone https://github.com/tu-usuario/tu-repositorio.git
   cd tu-repositorio

2. Ejecuta el programa  de tu preferecncia:

   ```bash
   python NonogramaPinguino.py

   ```bash
   python NonogramaCaballo.py
           
Esto lanzará el solucionador para cada nonograma y se abrirá la interfaz gráfica donde podrás ver la solución encontrada.

## Estructura del Proyecto
NonogramaPinguino.py:
Implementa el algoritmo genético, la extracción y comparación de secuencias, la búsqueda local para un nonograma con forma de pinguio, la interfaz es del tamaño del nonograma, 
además contiene las códificaciones de colores y las pistas correspondientes para su solución.

NonogramaCaballo.py:
Implementa el algoritmo genético, la extracción y comparación de secuencias, la búsqueda local para un nonograma con una imagen de un caballo, la interfaz es del tamaño del nonograma, 
además contiene las códificaciones de colores y las pistas correspondientes para su solución.

solucion.txt:
Archivo generado automáticamente que almacena la mejor solución encontrada.
