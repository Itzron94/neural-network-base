import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
from neural_network.core.network import NeuralNetwork
from tkinter import messagebox
import os

bg_color = '#3C3D37'
frame_color = '#181C14'
text_color = '#ECDFCC'


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocedor de Dígitos")
        self.root.configure(bg=bg_color)

        self.canvas_width = 280
        self.canvas_height = 280

        # Instrucciones
        label = tk.Label(self.root, text="Dibuja un dígito (0-9) y haz clic en 'Predecir'",
                         font=("Helvetica", 15, 'bold'), fg=text_color, bg=bg_color)
        label.pack(pady=10, ipadx=30)

        # Canvas para dibujar con un marco
        self.canvas_frame = tk.Frame(self.root, bg=frame_color, bd=10)
        self.canvas_frame.pack()
        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_width, height=self.canvas_height,
                                bg='white', highlightthickness=0)
        self.canvas.pack()

        # Botones
        btn_frame = tk.Frame(self.root, bg=bg_color)
        btn_frame.pack(pady=10)

        self.predict_button = tk.Button(
            btn_frame,
            text="Predecir",
            font=("Helvetica", 15),
            command=self.predict_digit,
            width=10
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(
            btn_frame,
            text="Borrar",
            font=("Helvetica", 15),
            command=self.clear_canvas,
            width=10
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Etiqueta para mostrar el resultado
        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 18, 'bold'), fg=text_color, bg=bg_color)
        self.result_label.pack(pady=10)

        # Contenedor para la imagen preprocesada (será visible después de la predicción)
        self.image_frame = tk.Frame(self.root, bg=bg_color)
        self.image_label = tk.Label(self.image_frame, bg=bg_color)
        self.image_label.pack()

        # Evento de dibujo
        self.canvas.bind("<B1-Motion>", self.draw)

        # Imagen para dibujar
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), 'white')
        self.draw_image = ImageDraw.Draw(self.image1)

        # Cargar la red neuronal entrenada
        weight_file = '../../resources/weights/mnist_weights_98.npz'
        if not os.path.exists(weight_file):
            messagebox.showerror("Error", f"No se encontró el archivo de pesos '{weight_file}'.")
            self.root.destroy()
        else:
            self.nn = NeuralNetwork.from_weights_file(weight_file)

        # Variable para almacenar la imagen preprocesada
        self.processed_image = None

    def draw(self, event):
        x = event.x
        y = event.y
        r = 10  # Radio del pincel
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.draw_image.ellipse([x - r, y - r, x + r, y + r], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_image.rectangle([0, 0, self.canvas_width, self.canvas_height], fill='white')
        # Limpiar la etiqueta de resultado y la imagen preprocesada
        self.result_label.config(text="")
        self.image_frame.pack_forget()
        self.processed_image = None

    def predict_digit(self):
        # Escalar proporcionalmente el dígito
        bbox = self.image1.getbbox()
        if not bbox:
            messagebox.showwarning("Advertencia", "Por favor, dibuja un dígito antes de predecir.")
            return

        digit_width = bbox[2] - bbox[0]
        digit_height = bbox[3] - bbox[1]
        max_dim = max(digit_width, digit_height)

        scale_factor = 28.0 / max_dim
        new_size = (int(digit_width * scale_factor), int(digit_height * scale_factor))
        resized_image = self.image1.crop(bbox).resize(new_size, Image.ANTIALIAS)
        resized_image = ImageOps.invert(resized_image)

        new_image = Image.new('L', (28, 28), color=0)
        upper_left = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
        new_image.paste(resized_image, upper_left)

        # Guardar la imagen preprocesada para mostrarla si se requiere
        self.processed_image = new_image

        # Convertir a arreglo NumPy y normalizar
        image_array = np.array(new_image).astype(np.float32) / 255.0
        image_array = image_array.reshape(1, 28 * 28)

        # Realizar predicción
        prediction = self.nn.predict(image_array)
        predicted_digit = np.argmax(prediction)
        probability = np.max(prediction) * 100

        # Mostrar el resultado en la etiqueta
        self.result_label.config(text=f"Dígito predicho: {predicted_digit}\nProbabilidad: {probability:.2f}%")

        # Mostrar la imagen preprocesada en la interfaz
        self.display_processed_image()

    def display_processed_image(self):
        """Muestra la imagen preprocesada en la interfaz."""
        if self.processed_image:
            # Redimensionar para que sea más visible (por ejemplo, 140x140 píxeles)
            display_image = self.processed_image.resize((140, 140), Image.NEAREST)
            processed_img_tk = ImageTk.PhotoImage(display_image)

            # Actualizar la etiqueta de imagen
            self.image_label.config(image=processed_img_tk)
            self.image_label.image = processed_img_tk  # Guardar referencia para evitar que la imagen se borre

            # Mostrar el frame de la imagen en la interfaz
            self.image_frame.pack(pady=10)


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()