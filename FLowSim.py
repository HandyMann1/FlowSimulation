import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class FlowSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Симуляция потока жидкости")
        self.root.geometry("1600x900")
        self.paused = False
        self.capilar_length = 10
        self.params = {'temperature': 20, 'viscosity': 0.001, 'density': 1000, 'min_width': 0.1, 'max_width': 1,
                       'particles_vx': 0.3, 'particles_vy': 0.1}
        self.particles_vx = None
        self.lam_y = None
        self.create_widgets()
        self.create_plot()
        self.animate()

    def create_widgets(self):
        control_frame = ttk.LabelFrame(self.root, text="Параметры")
        control_frame.pack(side=tk.LEFT, padx=10, pady=10)

        def add_label_entry(frame, text, row, param_key):
            ttk.Label(frame, text=text).grid(row=row, column=0, padx=5, pady=5)
            entry = ttk.Entry(frame, width=10)
            entry.insert(0, self.params[param_key])
            entry.grid(row=row, column=1)
            return entry

        self.temp_entry = add_label_entry(control_frame, "Температура (°C):", 0, 'temperature')
        self.visc_entry = add_label_entry(control_frame, "Вязкость (Па·с):", 1, 'viscosity')
        self.dens_entry = add_label_entry(control_frame, "Плотность (кг/м³):", 2, 'density')
        self.maxw_entry = add_label_entry(control_frame, "Ширина капилляра:", 3, 'max_width')
        self.vx_entry = add_label_entry(control_frame, "Скорость по X:", 4, 'particles_vx')
        self.vy_entry = add_label_entry(control_frame, "Скорость по Y:", 5, 'particles_vy')

        self.update_btn = ttk.Button(control_frame, text="Обновить параметры", command=self.update_params)
        self.update_btn.grid(row=6, column=0, columnspan=2, pady=10)
        self.pause_btn = ttk.Button(control_frame, text="Пауза", command=self.toggle_pause)
        self.pause_btn.grid(row=7, column=0, columnspan=2, pady=10)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn['text'] = "Продолжить" if self.paused else "Пауза"
        if not self.paused:
            self.animate()

    def create_plot(self):
        self.fig = plt.Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        self.particles_x = np.zeros(25)
        self.particles_y = np.linspace(0, self.params['max_width'], 25)
        self.update_params()

    def update_params(self):
        self.params['temperature'] = float(self.temp_entry.get())
        self.params['viscosity'] = float(self.visc_entry.get())
        self.params['density'] = float(self.dens_entry.get())
        self.params['max_width'] = float(self.maxw_entry.get())
        self.params['particles_vx'] = float(self.vx_entry.get())
        self.params['particles_vy'] = float(self.vy_entry.get())
        self.particles_x = np.zeros(25)
        self.particles_y = np.linspace(self.params['max_width']/43, self.params['max_width']-self.params['max_width']/43, 25)
        self.particles_vx = np.full(25, self.params['particles_vx'])
        self.particles_vy = np.zeros(25)

    def animate(self):
        if self.paused:
            return
        fluid = Fluid(temperature=self.params['temperature'], viscosity_neutral=self.params['viscosity'],
                      density=self.params['density'])
        diameter = self.params['min_width'] + (self.params['max_width'] - self.params['min_width']) * np.sin(np.pi / 2)
        re = fluid.reynolds_number(np.mean(self.particles_vx), diameter)
        self.ax.clear()
        self.ax.set_title(f'Поток жидкости (Re = {re:.0f})')
        self.ax.set_xlabel('Длина капилляра')
        self.ax.set_ylabel('Ширина капилляра')
        self.ax.set_xlim(0, self.capilar_length)
        self.ax.set_ylim(0, self.params['max_width'])
        self.ax.grid(True)
        self.ax.fill_between(np.linspace(0, self.capilar_length, 25), 0, diameter, alpha=0.2, color='blue')
        self.particles_x = (self.particles_x + self.particles_vx) % self.capilar_length
        for i in range(len(self.particles_x)):
            x = self.particles_x[i]
            y = self.particles_y[i]
            self.ax.arrow(0, y, x, 0, head_width=0.04, head_length=0.08, fc='black', ec='black', alpha=0.5)
        self.canvas.draw()
        self.root.after(50, self.animate)


class Fluid:
    def __init__(self, temperature, viscosity_neutral, density):
        self.temperature = temperature
        self.viscosity_neutral = viscosity_neutral
        self.density = density
        self.viscosity = self.calculate_viscosity()

    def calculate_viscosity(self):
        return self.viscosity_neutral

    def reynolds_number(self, velocity, diameter):
        return (self.density * velocity * diameter) / self.viscosity


if __name__ == "__main__":
    root = tk.Tk()
    app = FlowSimulator(root)
    root.mainloop()
