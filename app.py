import tkinter as tk
import torch
import numpy as np
from model import MLP

model = MLP()
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()

GRID_SIZE = 28
CELL_SIZE = 15  # smaller clean look

root = tk.Tk()
root.title("Paint")

canvas = tk.Canvas(
    root,
    width=GRID_SIZE * CELL_SIZE,
    height=GRID_SIZE * CELL_SIZE,
    bg="black"
)
canvas.pack()
grid_data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
rectangles = []

for row in range(GRID_SIZE):
    row_rects = []
    for col in range(GRID_SIZE):
        x1 = col * CELL_SIZE
        y1 = row * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE

        rect = canvas.create_rectangle(
            x1, y1, x2, y2,
            fill="black",
            outline="gray"
        )
        row_rects.append(rect)
    rectangles.append(row_rects)


def draw(event):
    col = event.x // CELL_SIZE
    row = event.y // CELL_SIZE

    if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
        grid_data[row, col] = 1.0
        canvas.itemconfig(rectangles[row][col], fill="white")
    predict()

canvas.bind("<B1-Motion>", draw)
canvas.bind("<Button-1>", draw)


def predict():
    img_tensor = torch.tensor(grid_data).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1).item()

    result_label.config(text=f"Prediction: {pred}")


def clear():
    global grid_data
    grid_data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            canvas.itemconfig(rectangles[row][col], fill="black")

    result_label.config(text="")

tk.Button(root, text="Clear", command=clear).pack()
result_label = tk.Label(root, text="", font=("Helvetica", 18))
result_label.pack()

root.mainloop()
