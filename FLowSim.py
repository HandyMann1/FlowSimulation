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
        self.params = {
            'temperature': 20,  # TODO добавить влияние температуры на поток
            'viscosity': 0.001,
            'density': 1000,
            'min_width': 0.1,
            'max_width': 1,
            'particles_vx': 0.3,
            'particles_vy': 0.1
        }
        self.particles_vx = None
        self.lam_y = None

        self.create_widgets()
        self.create_plot()
        self.animate()

    def create_widgets(self):  # инициализируем GUI
        control_frame = ttk.LabelFrame(self.root, text="Параметры")
        control_frame.pack(side=tk.LEFT, padx=10, pady=10)

        ttk.Label(control_frame, text="Температура (°C):").grid(row=0, column=0, padx=5, pady=5)
        self.temp_entry = ttk.Entry(control_frame, width=10)
        self.temp_entry.insert(0, self.params['temperature'])
        self.temp_entry.grid(row=0, column=1)

        ttk.Label(control_frame, text="Вязкость (Па·с):").grid(row=1, column=0, padx=5, pady=5)
        self.visc_entry = ttk.Entry(control_frame, width=10)
        self.visc_entry.insert(0, self.params['viscosity'])
        self.visc_entry.grid(row=1, column=1)

        ttk.Label(control_frame, text="Плотность (кг/м³):").grid(row=2, column=0, padx=5, pady=5)
        self.dens_entry = ttk.Entry(control_frame, width=10)
        self.dens_entry.insert(0, self.params['density'])
        self.dens_entry.grid(row=2, column=1)

        ttk.Label(control_frame, text="Ширина капиляра:").grid(row=3, column=0, padx=5, pady=5)
        self.maxw_entry = ttk.Entry(control_frame, width=10)
        self.maxw_entry.insert(0, self.params['max_width'])
        self.maxw_entry.grid(row=3, column=1)

        ttk.Label(control_frame, text="Скорость по X:").grid(row=4, column=0, padx=5, pady=5)
        self.vx_entry = ttk.Entry(control_frame, width=10)
        self.vx_entry.insert(0, self.params['particles_vx'])
        self.vx_entry.grid(row=4, column=1)

        ttk.Label(control_frame, text="Скорость по Y:").grid(row=5, column=0, padx=5, pady=5)
        self.vy_entry = ttk.Entry(control_frame, width=10)
        self.vy_entry.insert(0, self.params['particles_vy'])
        self.vy_entry.grid(row=5, column=1)

        self.update_btn = ttk.Button(control_frame, text="Обновить параметры",
                                     command=self.update_params)
        self.update_btn.grid(row=6, column=0, columnspan=2, pady=10)
        self.pause_btn = ttk.Button(control_frame, text="Пауза",
                                    command=self.toggle_pause)
        self.pause_btn.grid(row=7, column=0, columnspan=2, pady=10)

    def toggle_pause(self):  # поставить анимацию на паузу
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

        self.particles_x = np.linspace(0, self.capilar_length, 100)
        self.particles_y = np.random.uniform(0, self.params['max_width'], 100)
        self.particles_vy = np.random.uniform(-self.params['particles_vy'], self.params['particles_vy'], 100)
        self.update_params()

    def update_params(self):
        self.params['temperature'] = float(self.temp_entry.get())
        self.params['viscosity'] = float(self.visc_entry.get())
        self.params['density'] = float(self.dens_entry.get())
        self.params['max_width'] = float(self.maxw_entry.get())
        self.params['particles_vx'] = float(self.vx_entry.get())
        self.params['particles_vy'] = float(self.vy_entry.get())

        self.particles_x = np.linspace(0, self.capilar_length, 100)
        self.particles_y = np.random.uniform(0, self.params['max_width'], 100)

        self.particles_vx = np.random.uniform(0.1, self.params['particles_vx'], 100)
        self.particles_vy = np.random.uniform(-self.params['particles_vy'], self.params['particles_vy'], 100)
        self.lam_y = np.random.uniform(-self.params['max_width'], self.params['max_width'], 100)

    def animate(self):
        # TODO добавить векторные стрелочки направления
        if self.paused:  # пауза при нажатии
            return
        fluid = Fluid(  # инициализируем жидкость
            temperature=self.params['temperature'],
            viscosity_neutral=self.params['viscosity'],
            density=self.params['density']
        )

        diameter = self.params['min_width'] + (self.params['max_width'] - self.params['min_width']) * np.sin(np.pi / 2)
        re = fluid.reynolds_number(np.mean(self.particles_vx), diameter)  # считаем число Рейнольдса для жидкости
        flow_regime = "Ламинарный" if re < 2000 else "Турбулентный"  # выбор режима течения в зависимости от числа Рейнольдса
        color = 'blue' if flow_regime == "Ламинарный" else 'red'

        self.ax.clear()  # очищаем график и заново его заполняем
        self.ax.set_title(f'Поток жидкости (Re = {re:.1f}, Режим: {flow_regime})')
        self.ax.set_xlabel('Длина капиляра')
        self.ax.set_ylabel('Ширина капиляра')
        self.ax.set_xlim(0, self.capilar_length)
        self.ax.set_ylim(0, self.params['max_width'])
        self.ax.grid(True)
        self.ax.fill_between(np.linspace(0, self.capilar_length, 100), 0, diameter, alpha=0.2, color=color)
        # заполняем соответствующим цветом в зависимости от вида потока

        self.particles_x = (self.particles_x + self.particles_vx) % self.capilar_length

        if flow_regime == "Ламинарный":
            if not hasattr(self, 'lam_y'):
                self.lam_y = np.random.uniform(-diameter, diameter, 100)
            self.particles_y = self.lam_y
            self.particles_vx = np.random.uniform(0.1, self.params['particles_vx'], 100)


        else:

            self.particles_y = self.particles_y + self.particles_vy * np.random.uniform(-0.1, 0.1, 100)
            self.particles_vy = np.random.uniform(-self.params['particles_vy'], self.params['particles_vy'], 100)
            self.particles_y = np.clip(self.particles_y, -diameter, diameter)

        for i in range(len(self.particles_x)):
            x = self.particles_x[i]
            y = self.particles_y[i]
            vx = self.particles_vx[i]
            vy = self.particles_vy[i]

            length = np.sqrt(vx ** 2 + vy ** 2)
            if length > 0:
                norm_vx = vx / length * 0.1
                norm_vy = vy / length * 0.1
            else:
                norm_vx = 0
                norm_vy = 0

            self.ax.arrow(x, y, norm_vx, norm_vy,
                          head_width=0.005, head_length=0.01,
                          color='black', alpha=0.5)

        self.ax.scatter(self.particles_x, self.particles_y, s=self.capilar_length, color='black')
        self.canvas.draw()
        self.root.after(50, self.animate)


class Fluid:  # характеристики жидкости + находим число Рейнольдса
    def __init__(self, temperature, viscosity_neutral, density):
        self.temperature = temperature
        self.viscosity_neutral = viscosity_neutral
        self.density = density
        self.viscosity = self.calculate_viscosity()

    def calculate_viscosity(self):
        return self.viscosity_neutral  # TODO доделать вычисление по формуле

    def reynolds_number(self, velocity, diameter):
        return (self.density * velocity * diameter) / self.viscosity


if __name__ == "__main__":
    root = tk.Tk()
    app = FlowSimulator(root)
    root.mainloop()
