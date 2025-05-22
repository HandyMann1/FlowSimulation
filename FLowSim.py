import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch

class FlowSimulator:
    """
    Класс для симуляции потока жидкости в капилляре с визуализацией и интерактивным управлением параметрами.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Симуляция потока жидкости")
        self.root.geometry("1600x900")
        self.paused = False
        self.params = {
            'temperature': 293,
            'viscosity': 0.2,
            'density': 1000,
            'min_width': 0.01,
            'max_width': 0.1,
            'capillary_length': 1000,
            'particles_vx': 0.3,
            'Ea': 1000,
            'T0': 273,
            'R': 8.314,
            'pressure_start': 1013250,
            'pressure_end': 100000
        }
        self.viscosity = self.calculate_viscosity()
        self.animation_reset = False
        self.arrow_patches = []

        self.create_widgets()
        self.create_plot()
        self.particles_vx = self.calculate_particle_speed()
        self.create_viscosity_info_label()
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

        self.temp_entry = add_label_entry(control_frame, "Температура (K):", 0, 'temperature')
        self.visc_entry = add_label_entry(control_frame, "Начальная вязкость (Па·с):", 1, 'viscosity')
        self.dens_entry = add_label_entry(control_frame, "Плотность (кг/м³):", 2, 'density')
        self.maxw_entry = add_label_entry(control_frame, "Ширина капилляра, м:", 3, 'max_width')
        self.cap_len_entry = add_label_entry(control_frame, "Длина капилляра, м:", 4, "capillary_length")
        self.Ea_entry = add_label_entry(control_frame, "Активационная энергия (Дж/моль):", 7, 'Ea')
        self.T0_entry = add_label_entry(control_frame, "Базовая температура (К):", 8, 'T0')
        self.p_start_entry = add_label_entry(control_frame, "Начальное давление (Па):", 10, 'pressure_start')
        self.p_end_entry = add_label_entry(control_frame, "Конечное давление (Па):", 11, 'pressure_end')

        self.update_btn = ttk.Button(control_frame, text="Обновить параметры", command=self.update_params)
        self.update_btn.grid(row=12, column=0, columnspan=2, pady=10)
        self.pause_btn = ttk.Button(control_frame, text="Пауза", command=self.toggle_pause)
        self.pause_btn.grid(row=13, column=0, columnspan=2, pady=10)

        # Метка для вывода информации о вязкости
        self.visc_info_label = ttk.Label(control_frame, text="", wraplength=220, justify="left")
        self.visc_info_label.grid(row=14, column=0, columnspan=2, pady=10)

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
        self.params['capillary_length'] = float(self.cap_len_entry.get())
        self.params['Ea'] = float(self.Ea_entry.get())
        self.params['T0'] = float(self.T0_entry.get())
        self.params['pressure_start'] = float(self.p_start_entry.get())
        self.params['pressure_end'] = float(self.p_end_entry.get())
        self.particles_x = np.zeros(25)
        self.particles_y = np.linspace(
            self.params['max_width'] / 25,
            self.params['max_width'] - self.params['max_width'] / 25, 25
        )
        self.viscosity = self.calculate_viscosity()
        self.particles_vx = self.calculate_particle_speed()
        self.update_viscosity_info()

    def calculate_particle_speed(self):
        press_diff = self.params["pressure_start"] - self.params["pressure_end"]
        denominator = 4 * self.viscosity * self.params["capillary_length"]
        R = self.params["max_width"] / 2
        distance_from_center = np.abs(self.particles_y - R)
        r_sqr_diff = np.maximum(0, R ** 2 - distance_from_center ** 2)
        return (press_diff / denominator) * r_sqr_diff

    def animate(self):
        if self.paused:
            return

        avg_velocity = np.mean(self.particles_vx)
        re = self.reynolds_number(avg_velocity)

        if re < 1:
            self.particles_vx.fill(avg_velocity)
        else:
            self.particles_vx = self.calculate_particle_speed()

        re = self.reynolds_number(np.mean(self.particles_vx))
        self.ax.clear()
        self.ax.set_title(f'Поток жидкости (Re = {re:.0f})')
        self.ax.set_xlabel('Длина капилляра, м')
        self.ax.set_ylabel('Ширина капилляра, м')
        self.ax.set_xlim(0, self.params['capillary_length'])
        self.ax.set_ylim(0, self.params['max_width'])
        self.ax.grid(True)
        self.ax.fill_between(
            np.linspace(0, self.params['capillary_length'], 25),
            0, self.params["max_width"],
            alpha=0.2, color='blue'
        )

        self.particles_x += self.particles_vx

        if np.any(self.particles_x >= self.params['capillary_length']):
            self.particles_x = np.zeros(25)
            self.animation_reset = True

        # Удаляем старые стрелки
        for arrow in getattr(self, 'arrow_patches', []):
            arrow.remove()
        self.arrow_patches = []

        # Добавляем новые стрелки с фиксированным размером головки (mutation_scale)
        if self.animation_reset:
            self.animation_reset = False
            for i in range(len(self.particles_x)):
                y = self.particles_y[i]
                arrow = FancyArrowPatch(
                    (0, y), (0.01, y),
                    mutation_scale=20,  # размер головки в пикселях
                    color='black', alpha=0.5, linewidth=2
                )
                self.ax.add_patch(arrow)
                self.arrow_patches.append(arrow)
        else:
            for i in range(len(self.particles_x)):
                x = self.particles_x[i]
                y = self.particles_y[i]
                arrow = FancyArrowPatch(
                    (0, y), (x, y),
                    mutation_scale=20,  # размер головки в пикселях
                    color='black', alpha=0.5, linewidth=2
                )
                self.ax.add_patch(arrow)
                self.arrow_patches.append(arrow)

        self.canvas.draw()
        self.root.after(10, self.animate)

    def calculate_viscosity(self):
        return self.params["viscosity"] * np.exp(
            -(self.params["Ea"] / (self.params["R"] * self.params["T0"] ** 2)) *
            (self.params["temperature"] - self.params["T0"])
        )

    def reynolds_number(self, velocity):
        return (self.params["density"] * velocity * self.params["max_width"]) / self.viscosity

    def create_viscosity_info_label(self):
        self.update_viscosity_info()

    def update_viscosity_info(self):
        avg_velocity = np.mean(self.particles_vx)
        Re = self.reynolds_number(avg_velocity)

        if Re > 1:
            msg = f"Число Рейнольдса (Re) = {Re:.2f}. Вязкостью можно пренебречь, так как Re > 1."
        else:
            msg = f"Число Рейнольдса (Re) = {Re:.10f}. Вязкость существенно влияет, пренебрегать нельзя."

        self.visc_info_label.config(text=msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = FlowSimulator(root)
    root.mainloop()
