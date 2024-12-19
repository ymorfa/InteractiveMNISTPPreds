import tkinter as tk
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk
from draw_tools import DrawNN
from datetime import datetime
import threading


class MNISTDrawer:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("MNIST Drawer")

        # Grid and drawing setup
        self.grid_size = 28
        self.cell_size = 20
        self.input = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.last_input = np.copy(self.input)  # To track changes and avoid redundant predictions

        # Flags and threading
        self.prediction_thread = None
        self.prediction_running = False

        # Create main frames
        self.canvas = tk.Canvas(self.root, width=self.grid_size * self.cell_size,
                                 height=self.grid_size * self.cell_size, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.paint)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(side=tk.RIGHT, padx=10, pady=10)

        self.create_menu()
        self.refresh_canvas()
        self.update_output_image()

    def create_menu(self):
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.reset)

    def paint(self, event):
        x, y = event.x, event.y
        row, col = y // self.cell_size, x // self.cell_size
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            self.update_cell(row, col, 0.8)
            self.update_neighbors(row, col, 0.15)
            self.clip_input()
            #self.refresh_canvas()
            self.root.after_idle(self.update_cell_canvas, row, col)  # Update only affected cells

    def update_cell(self, row, col, value):
        self.input[row, col] += value

    def update_neighbors(self, row, col, value):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                self.input[nr, nc] += value

    def clip_input(self):
        self.input = np.clip(self.input, 0, 1)

    def refresh_canvas(self):
        self.canvas.delete("all")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.update_cell_canvas(i, j)

    def update_cell_canvas(self, row, col):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                intensity = int((1 - self.input[nr, nc]) * 255)
                color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"

                x1, y1 = nc * self.cell_size, nr * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
            
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def update_output_image(self):
        if self.prediction_running or np.array_equal(self.input, self.last_input):
            self.root.after(800, self.update_output_image)
            return  # Skip prediction if nothing has changed or a prediction is running

        self.last_input = np.copy(self.input)  # Update the last input state
        self.prediction_running = True

        def predict():
            try:
                input_array = np.expand_dims(self.input, axis=0)
                network = DrawNN(self.model, input_array)
                img_stream = network.draw()
                img = Image.open(img_stream)
                self.tk_image = ImageTk.PhotoImage(img)
                self.image_label.configure(image=self.tk_image)
            except Exception as e:
                print(f"Error during prediction: {e}")
            finally:
                self.prediction_running = False

        # Run prediction in a separate thread
        self.prediction_thread = threading.Thread(target=predict)
        self.prediction_thread.start()

        # Schedule the next update
        self.root.after(800, self.update_output_image)

    def reset(self):
        self.input = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.refresh_canvas()
        self.prediction_running = False
        self.update_output_image()


if __name__ == "__main__":
    # Load your model here
    model = load_model('./models/2L25N_softmax_model.keras')

    root = tk.Tk()
    app = MNISTDrawer(root, model)
    root.mainloop()
