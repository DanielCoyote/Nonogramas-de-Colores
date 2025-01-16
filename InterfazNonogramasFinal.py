import tkinter as tk

class NonogramGame:
    def __init__(self, root, nonogram):
        self.root = root
        self.root.title("Nonograma")

        # Establecer color de fondo
        self.root.configure(bg="#f0f0f0")

        # Crear un marco para el contenido
        self.frame = tk.Frame(self.root, bg="#ffffff", padx=20, pady=20)
        self.frame.pack(pady=20)

        # Título
        self.titulo_label = tk.Label(self.frame, text="Mejor Solución Encontrada", font=("Arial", 16, "bold"), bg="#ffffff")
        self.titulo_label.pack(pady=10)

        # Establecer el tamaño de las celdas y las dimensiones de la cuadrícula
        self.cell_size = 30  # Ajusta el tamaño de la celda
        self.board_width = 17  # Número de columnas
        self.board_height = 20  # Número de filas

        # Crear un canvas donde se dibujará la cuadrícula
        self.canvas = tk.Canvas(self.frame, width=self.board_width * self.cell_size, 
                                height=self.board_height * self.cell_size, bg="white")
        self.canvas.pack(padx=10, pady=10)

        # Mostrar la cuadrícula
        self.nonogram = nonogram
        self.board = nonogram["r"]
        self.clues = nonogram["c"]
        self.solution = nonogram.get("solution", None)  # Obtener la solución del nonograma
        self.cells = []  # Lista para almacenar las celdas del canvas
        self.draw_grid()

        # Botones
        self.create_buttons()

    def create_buttons(self):
        """ Crear los botones de la interfaz """
        button_data = [
            ("Mostrar Mejor Solución", self.mostrar_mejor_solucion, "#FFB3BA"),
            ("Salir", self.root.quit, "#FFDFBA")
        ]
        
        for text, command, color in button_data:
            button = tk.Button(self.frame, text=text, command=command, font=("Arial", 14), bg=color, fg="#5B2E4E", padx=10, pady=5)
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
            print("No se encontró una solución para mostrar.")
            return

        # Mapeo de números a colores
        numero_a_color = {
            0: "#FFFFFF",  # Blanco
            1: "#00FFFF",  # Celeste
            2: "#000000",  # Negro
            3: "#008000",  # Verde
            4: "#FFA500"   # Naranja
        }

        for i in range(self.board_height):
            for j in range(self.board_width):
                color_value = self.solution[i][j]
                color = numero_a_color.get(color_value, "#FFFFFF")  # Usa blanco como color predeterminado si no encuentra el valor
                self.canvas.itemconfig(self.cells[i][j], fill=color)

    def actualizar_solucion(self, nueva_solucion):
        """ Actualiza la solución del nonograma """
        # Verificar que la nueva solución contenga solo valores válidos (0, 1, 2, 3, 4)
        for fila in nueva_solucion:
            for valor in fila:
                if valor not in {0, 1, 2, 3, 4}:  # Verifica que solo haya valores permitidos
                    print(f"Advertencia: valor no válido {valor} encontrado en la solución.")
                    return

        # Si la solución es válida, la asignamos
        self.solution = nueva_solucion
        print("Solución actualizada.")
        self.mostrar_mejor_solucion()

        # Verificar si la solución no está vacía antes de guardar
        if self.solution and any(self.solution):  # Verifica si la solución no está vacía
            self.guardar_solucion()
        else:
            print("La solución está vacía y no se guardará.")


    def guardar_solucion(self):
        """ Guarda la solución transformada en un archivo de texto """
        if not self.solution:
            print("No hay solución para guardar.")
            return

        # Mapeo de números a colores
        numero_a_color = {
            0: "Blanco",   # Blanco
            1: "Celeste",  # Celeste
            2: "Negro",    # Negro
            3: "Verde",    # Verde
            4: "Naranja"   # Naranja
        }

        # Convertir la solución en colores
        solucion_colores = [numero_a_color.get(self.solution[i][j], "Blanco") for i in range(self.board_height) for j in range(self.board_width)]

        # Guardar la solución en un archivo
        ruta_guardado = 'C:\\Users\\mely_\\Downloads\\Nonograma a color\\solucion_transformada.txt'  # Ruta del archivo donde se guardará la solución
        try:
            with open(ruta_guardado, 'w') as archivo:
                archivo.write(" ".join(solucion_colores))
            print(f"Solución guardada en {ruta_guardado}")
        except Exception as e:
            print(f"Error al guardar la solución: {e}")

def cargar_solucion_desde_archivo(ruta_archivo):
    """ Carga la solución desde un archivo de texto y la convierte en una matriz de números """
    with open(ruta_archivo, 'r') as archivo:
        colores = archivo.read().split()  # Leer todos los colores

    # Verificar si la lista de colores está vacía
    if not colores:
        print("Advertencia: El archivo está vacío o no contiene datos válidos.")
        return None  # Devuelve None si no hay datos válidos

    # Mapeo de colores a números
    color_a_numero = {
        "Blanco": 0,   # Blanco -> 0
        "Celeste": 1,   # Celeste -> 1
        "Negro": 2,     # Negro -> 2      
        "Verde": 3,     # Verde -> 3
        "Naranja": 4    # Naranja -> 4
    }

    solucion = [color_a_numero.get(color, 0) for color in colores]  # Convertir colores a números

    # Verificar que la cantidad de valores en la solución coincida con el tamaño del tablero
    board_width = 17  # Ajusta esto a las dimensiones correctas del tablero
    if len(solucion) != board_width * 20:  # Asegurarse de que la solución tenga el tamaño correcto (20x17 en este caso)
        print("Advertencia: El tamaño de la solución no coincide con el tamaño del tablero.")
    
    solucion_matriz = [solucion[i:i + board_width] for i in range(0, len(solucion), board_width)]
    return solucion_matriz


if __name__ == "__main__":
    # Cargar el nonograma desde un archivo
    ruta_archivo = 'C:\\Users\\mely_\\Downloads\\Nonograma a color\\solucion.txt'  # Ruta del archivo de solución
    mejor_solucion = cargar_solucion_desde_archivo(ruta_archivo)

    nonogramas = [
        {
            "r": [[0]*17 for _ in range(20)],  # Tablero vacío (debe ser completado por el algoritmo)
            "c": [[0]*17 for _ in range(20)],  # Pistas de columnas (vacías por ahora)
            "solution": mejor_solucion  # Asignar la mejor solución
        }
    ]

    root = tk.Tk()
    nonogram_game = NonogramGame(root, nonogramas[0])  # Usando el primer nonograma

    # Llamar a actualizar_solucion con la solución cargada
    nonogram_game.actualizar_solucion(mejor_solucion)  # Actualizar la interfaz con la mejor solución

    root.mainloop()
