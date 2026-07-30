import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.special import jv, yv, iv, kv
import sympy as sp
import time
import warnings
import threading
from queue import Queue
import sys
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
class FiberModeSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAF-Solver")
        self.root.geometry("1600x900")
        self.default_params = {
            'a1': 10e-6,
            'a2': 19e-6,
            'a3': 22e-6,
            'n1': 1.4512,
            'n2': 1.45,
            'n3': 1.4512,
            'nclad': 1.45,
            'wavelength': 1.08e-6
        }
        self.current_params = self.default_params.copy()
        self.results = {}
        self.computation_queue = Queue()
        self.calculating = False
        self.calculation_thread = None
        self.setup_gui()
        self.update_refractive_index_plot()
    def setup_gui(self):
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_panel = ttk.Frame(main_container, width=600)
        main_container.add(left_panel, weight=1)
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=3)
        left_panel.pack_propagate(False)
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
    def setup_left_panel(self, parent):
        title_label = ttk.Label(parent, text="RAF-Solver",
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Parameters")
        self.setup_parameters_tab(params_frame)
        modes_frame = ttk.Frame(notebook)
        notebook.add(modes_frame, text="Modes")
        self.setup_modes_tab(modes_frame)
        self.setup_control_buttons(parent)
    def setup_parameters_tab(self, parent):
        canvas = tk.Canvas(parent, borderwidth=0, width=380)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        params = [
            ("Core Radius (a1, μm):", "a1", 10e-6, "Core radius in meters"),
            ("Trench Radius (a2, μm):", "a2", 19e-6, "Trench radius in meters"),
            ("Ring Radius (a3, μm):", "a3", 22e-6, "Ring radius in meters"),
            ("Core Index (n1):", "n1", 1.4512, "Core refractive index"),
            ("Trench Index (n2):", "n2", 1.45, "Trench refractive index"),
            ("Ring Index (n3):", "n3", 1.4512, "Ring refractive index"),
            ("Cladding Index (nclad):", "nclad", 1.45, "Cladding refractive index"),
            ("Wavelength (μm):", "wavelength", 1.08e-6, "Wavelength in meters"),
        ]
        self.param_entries = {}
        for i, (label_text, param_name, default_value, tooltip) in enumerate(params):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, padx=10, pady=5)
            label = ttk.Label(frame, text=label_text, width=25, font=('Arial', 9))
            label.pack(side=tk.LEFT, padx=(0, 5))
            entry = ttk.Entry(frame, width=20, font=('Arial', 9))
            entry.insert(0, str(default_value * 1e6 if param_name in ['a1', 'a2', 'a3', 'wavelength'] else default_value))
            entry.pack(side=tk.LEFT)
            self.create_tooltip(label, tooltip)
            entry.bind('<KeyRelease>', lambda e, p=param_name: self.on_parameter_change(p))
            self.param_entries[param_name] = entry
    def setup_modes_tab(self, parent):
        ttk.Label(parent, text="Select Modes to Compute:",
                 font=('Arial', 10, 'bold')).pack(pady=8)
        self.mode_vars = {}
        modes = [
            ("LP01 (m=0, even)", "LP01"),
            ("LP11e (m=1, even)", "LP11e"),
            ("LP11o (m=1, odd)", "LP11o"),
            ("LP21e (m=2, even)", "LP21e"),
            ("LP21o (m=2, odd)", "LP21o"),
            ("LP02 (m=0, even)", "LP02"),
            ("LP31e (m=3, even)", "LP31e"),
            ("LP31o (m=3, odd)", "LP31o"),
            ("LP12e (m=1, even)", "LP12e"),
            ("LP12o (m=1, odd)", "LP12o"),
        ]
        modes_frame = ttk.Frame(parent)
        modes_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        for i, (mode_text, mode_key) in enumerate(modes):
            row = i // 2
            col = i % 2
            var = tk.BooleanVar(value=True if i < 6 else False)
            cb = ttk.Checkbutton(modes_frame, text=mode_text, variable=var, width=15)
            cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            self.mode_vars[mode_key] = var
        modes_frame.grid_columnconfigure(0, weight=1)
        modes_frame.grid_columnconfigure(1, weight=1)
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="Select All",
                  command=self.select_all_modes, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Select None",
                  command=self.deselect_all_modes, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="First 6",
                  command=self.select_first_six_modes, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Last 4",
                  command=self.select_last_four_modes, width=12).pack(side=tk.LEFT, padx=2)
    def setup_control_buttons(self, parent):
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=5, pady=8)
        self.calc_btn = ttk.Button(btn_frame, text="Calculate Modes",
                                  command=self.calculate_modes, width=14)
        self.calc_btn.pack(side=tk.LEFT, padx=2)
        reset_btn = ttk.Button(btn_frame, text="Reset Parameters",
                              command=self.reset_parameters, width=14)
        reset_btn.pack(side=tk.LEFT, padx=2)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(btn_frame, variable=self.progress_var,
                                           maximum=100, length=120)
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(btn_frame, text="Ready", width=15, font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT, padx=2)
    def setup_right_panel(self, parent):
        self.plot_notebook = ttk.Notebook(parent)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ref_index_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.ref_index_frame, text="Refractive Index Profile")
        self.results_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.results_frame, text="Mode Results")
        self.table_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.table_frame, text="Results Table")
        self.setup_ref_index_plot()
        self.setup_results_display()
        self.setup_results_table()
    def setup_ref_index_plot(self):
        self.ref_index_fig = Figure(figsize=(10, 6), dpi=100)
        self.ref_index_ax = self.ref_index_fig.add_subplot(111)
        self.ref_index_fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        self.ref_index_canvas = FigureCanvasTkAgg(self.ref_index_fig, self.ref_index_frame)
        self.ref_index_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        toolbar = NavigationToolbar2Tk(self.ref_index_canvas, self.ref_index_frame)
        toolbar.update()
        self.ref_index_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def setup_results_display(self):
        results_canvas = tk.Canvas(self.results_frame, borderwidth=0)
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical",
                                 command=results_canvas.yview)
        self.results_inner_frame = ttk.Frame(results_canvas)
        self.results_inner_frame.bind(
            "<Configure>",
            lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all"))
        )
        results_canvas.create_window((0, 0), window=self.results_inner_frame, anchor="nw")
        results_canvas.configure(yscrollcommand=scrollbar.set)
        results_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.results_label = ttk.Label(self.results_inner_frame,
                                      text="No results yet. Click 'Calculate Modes' to start.",
                                      font=('Arial', 12))
        self.results_label.pack(pady=50)
        self.result_plots = {}
    def setup_results_table(self):
        table_container = ttk.Frame(self.table_frame)
        table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        columns = ("Mode", "neff", "fval", "Iterations")
        self.results_tree = ttk.Treeview(table_container, columns=columns, show="headings", height=20)
        col_widths = {"Mode": 80, "neff": 120, "fval": 100, "Iterations": 80}
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=col_widths.get(col, 100), anchor=tk.CENTER)
        scrollbar = ttk.Scrollbar(table_container, orient="vertical",
                                 command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        export_btn = ttk.Button(self.table_frame, text="Export to CSV",
                               command=self.export_results)
        export_btn.pack(pady=10)
    def create_tooltip(self, widget, text):
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, background="lightyellow",
                             relief="solid", borderwidth=1, font=('Arial', 9))
            label.pack()
            def hide_tooltip():
                tooltip.destroy()
            widget.bind("<Leave>", lambda e: hide_tooltip())
        widget.bind("<Enter>", show_tooltip)
    def on_parameter_change(self, param_name):
        try:
            value_str = self.param_entries[param_name].get()
            value = float(value_str)
            if param_name in ['a1', 'a2', 'a3', 'wavelength']:
                value = value * 1e-6
            self.current_params[param_name] = value
            self.update_refractive_index_plot()
        except ValueError:
            pass
    def update_refractive_index_plot(self):
        self.ref_index_ax.clear()
        a1 = self.current_params['a1']
        a2 = self.current_params['a2']
        a3 = self.current_params['a3']
        n1 = self.current_params['n1']
        n2 = self.current_params['n2']
        n3 = self.current_params['n3']
        nclad = self.current_params['nclad']
        r_max = a3 * 2.5
        r = np.linspace(0, r_max, 1000)
        n = np.ones_like(r) * nclad
        n[r <= a1] = n1
        n[(r > a1) & (r <= a2)] = n2
        n[(r > a2) & (r <= a3)] = n3
        r_um = r * 1e6
        self.ref_index_ax.plot(r_um, n, 'b-', linewidth=2)
        self.ref_index_ax.set_xlabel('Radial Position (μm)', fontsize=10)
        self.ref_index_ax.set_ylabel('Refractive Index', fontsize=10)
        self.ref_index_ax.set_title('1D Refractive Index Profile', fontsize=11, fontweight='bold')
        self.ref_index_ax.grid(True, alpha=0.3)
        self.ref_index_ax.axvline(x=a1*1e6, color='g', linestyle='--', alpha=0.7,
                                 label=f'Core: {a1*1e6:.1f} μm')
        self.ref_index_ax.axvline(x=a2*1e6, color='r', linestyle='--', alpha=0.7,
                                 label=f'Trench: {a2*1e6:.1f} μm')
        self.ref_index_ax.axvline(x=a3*1e6, color='orange', linestyle='--', alpha=0.7,
                                 label=f'Ring: {a3*1e6:.1f} μm')
        self.ref_index_ax.legend(loc='upper right', fontsize=9)
        self.ref_index_ax.set_xlim(0, r_max * 1e6)
        self.ref_index_fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        self.ref_index_canvas.draw()
    def select_all_modes(self):
        for var in self.mode_vars.values():
            var.set(True)
    def deselect_all_modes(self):
        for var in self.mode_vars.values():
            var.set(False)
    def select_first_six_modes(self):
        first_six = ['LP01', 'LP11e', 'LP11o', 'LP21e', 'LP21o', 'LP02']
        for mode, var in self.mode_vars.items():
            var.set(mode in first_six)
    def select_last_four_modes(self):
        last_four = ['LP31e', 'LP31o', 'LP12e', 'LP12o']
        for mode, var in self.mode_vars.items():
            var.set(mode in last_four)
    def calculate_modes(self):
        if self.calculating:
            messagebox.showwarning("Calculation in Progress",
                                 "Please wait for current calculation to finish.")
            return
        selected_modes = [mode for mode, var in self.mode_vars.items() if var.get()]
        if not selected_modes:
            messagebox.showwarning("No Modes Selected",
                                 "Please select at least one mode to calculate.")
            return
        self.calc_btn.config(state='disabled')
        self.status_label.config(text="Calculating...")
        self.progress_var.set(0)
        self.calculating = True
        self.clear_results_display()
        self.calculation_thread = threading.Thread(
            target=self.calculate_modes_thread,
            args=(selected_modes,)
        )
        self.calculation_thread.daemon = True
        self.calculation_thread.start()
        self.monitor_calculation()
    def calculate_modes_thread(self, selected_modes):
        try:
            params = self.current_params.copy()
            wavelength = params['wavelength']
            k0 = 2 * np.pi / wavelength
            yita2 = params['a2'] / params['a1']
            yita3 = params['a3'] / params['a1']
            base_parameters = [k0, params['n1'], params['n2'], params['n3'],
                              params['nclad'], params['a1'], yita2, yita3]
            self.results = {}
            if 'LP02' in selected_modes and 'LP01' not in selected_modes:
                try:
                    m_lp01, orientation_lp01 = 0, 'even'
                    initial_neff_lp01 = params['n1']
                    lp01_neff, lp01_fval, lp01_iterations = self.solve_neff_for_mode(
                        m_lp01, initial_neff_lp01, base_parameters, "LP01"
                    )
                    self.lp01_temp_neff = lp01_neff
                except Exception as e:
                    self.lp01_temp_neff = params['n1']
                    print(f"Failed to calculate LP01 as initial value for LP02: {e}")
            else:
                self.lp01_temp_neff = None
            if ('LP12e' in selected_modes or 'LP12o' in selected_modes) and 'LP11e' not in selected_modes and 'LP11o' not in selected_modes:
                try:
                    m_lp11, orientation_lp11 = 1, 'even'
                    initial_neff_lp11 = params['n1']
                    lp11_neff, lp11_fval, lp11_iterations = self.solve_neff_for_mode(
                        m_lp11, initial_neff_lp11, base_parameters, "LP11e"
                    )
                    self.lp11_temp_neff = lp11_neff
                except Exception as e:
                    self.lp11_temp_neff = params['n1'] - 0.0001
                    print(f"Failed to calculate LP11 as initial value for LP12: {e}")
            else:
                self.lp11_temp_neff = None
            total_modes = len(selected_modes)
            for i, mode_name in enumerate(selected_modes):
                self.computation_queue.put(('progress', (i+1)/total_modes * 100))
                if mode_name == 'LP01':
                    m, orientation = 0, 'even'
                    initial_neff = params['n1']
                elif mode_name == 'LP11e':
                    m, orientation = 1, 'even'
                    initial_neff = params['n1']
                elif mode_name == 'LP11o':
                    m, orientation = 1, 'odd'
                    initial_neff = params['n1']
                elif mode_name == 'LP21e':
                    m, orientation = 2, 'even'
                    initial_neff = params['n1']
                elif mode_name == 'LP21o':
                    m, orientation = 2, 'odd'
                    initial_neff = params['n1']
                elif mode_name == 'LP02':
                    m, orientation = 0, 'even'
                    if hasattr(self, 'lp01_temp_neff') and self.lp01_temp_neff is not None:
                        initial_neff = self.lp01_temp_neff - 0.0001
                    else:
                        if 'LP01' in self.results:
                            initial_neff = self.results['LP01']['neff'] - 0.0001
                        else:
                            initial_neff = params['n1'] - 0.0001
                elif mode_name == 'LP31e':
                    m, orientation = 3, 'even'
                    initial_neff = params['n1']
                elif mode_name == 'LP31o':
                    m, orientation = 3, 'odd'
                    initial_neff = params['n1']
                elif mode_name == 'LP12e':
                    m, orientation = 1, 'even'
                    if hasattr(self, 'lp11_temp_neff') and self.lp11_temp_neff is not None:
                        initial_neff = self.lp11_temp_neff - 0.00001
                    else:
                        if 'LP11e' in self.results:
                            initial_neff = self.results['LP11e']['neff'] - 0.00001
                        elif 'LP11o' in self.results:
                            initial_neff = self.results['LP11o']['neff'] - 0.00001
                        else:
                            initial_neff = params['n1'] - 0.0001
                elif mode_name == 'LP12o':
                    m, orientation = 1, 'odd'
                    if hasattr(self, 'lp11_temp_neff') and self.lp11_temp_neff is not None:
                        initial_neff = self.lp11_temp_neff - 0.00001
                    else:
                        if 'LP11e' in self.results:
                            initial_neff = self.results['LP11e']['neff'] - 0.00001
                        elif 'LP11o' in self.results:
                            initial_neff = self.results['LP11o']['neff'] - 0.00001
                        else:
                            initial_neff = params['n1'] - 0.0001
                neff, fval, iterations = self.solve_neff_for_mode(
                    m, initial_neff, base_parameters, mode_name
                )
                coefficients = self.compute_mode_coefficients(neff, m, base_parameters)
                E, I = self.compute_mode_field(m, orientation, neff, coefficients,
                                              base_parameters)
                self.results[mode_name] = {
                    'neff': neff,
                    'fval': fval,
                    'iterations': iterations,
                    'coefficients': coefficients,
                    'E': E,
                    'I': I,
                    'm': m,
                    'orientation': orientation
                }
                self.computation_queue.put(('result', (mode_name, E, I, neff)))
            if hasattr(self, 'lp01_temp_neff'):
                del self.lp01_temp_neff
            if hasattr(self, 'lp11_temp_neff'):
                del self.lp11_temp_neff
            self.computation_queue.put(('complete', None))
        except Exception as e:
            self.computation_queue.put(('error', str(e)))
    def monitor_calculation(self):
        try:
            while not self.computation_queue.empty():
                msg_type, data = self.computation_queue.get_nowait()
                if msg_type == 'progress':
                    self.progress_var.set(data)
                elif msg_type == 'result':
                    mode_name, E, I, neff = data
                    self.display_mode_result(mode_name, E, I, neff)
                elif msg_type == 'complete':
                    self.calculation_complete()
                    return
                elif msg_type == 'error':
                    self.calculation_error(data)
                    return
            self.root.after(100, self.monitor_calculation)
        except:
            self.root.after(100, self.monitor_calculation)
    def calculation_complete(self):
        self.calculating = False
        self.calc_btn.config(state='normal')
        self.status_label.config(text="Complete")
        self.progress_var.set(100)
        self.update_results_table()
        self.plot_notebook.select(self.results_frame)
        messagebox.showinfo("Calculation Complete",
                          f"Successfully calculated {len(self.results)} modes.")
    def calculation_error(self, error_msg):
        self.calculating = False
        self.calc_btn.config(state='normal')
        self.status_label.config(text="Error")
        messagebox.showerror("Calculation Error", f"Error during calculation:\n{error_msg}")
    def clear_results_display(self):
        for widget in self.results_inner_frame.winfo_children():
            widget.destroy()
        self.results_label = ttk.Label(self.results_inner_frame,
                                      text="Calculating...",
                                      font=('Arial', 12))
        self.results_label.pack(pady=50)
    def display_mode_result(self, mode_name, E, I, neff):
        if self.results_label.winfo_exists():
            self.results_label.destroy()
        mode_frame = ttk.LabelFrame(self.results_inner_frame, text=f"{mode_name} - neff: {neff:.6f}")
        mode_frame.pack(fill=tk.X, padx=10, pady=5, ipadx=5, ipady=5)
        fig = Figure(figsize=(7.5, 2.5), dpi=80)
        fig.subplots_adjust(left=0.08, right=0.96, top=0.9, bottom=0.15, wspace=0.3)
        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(E.real, cmap='RdBu',
                        extent=[-self.current_params['a3']*2*1e6,
                                self.current_params['a3']*2*1e6,
                                -self.current_params['a3']*2*1e6,
                                self.current_params['a3']*2*1e6])
        ax1.set_title('Electric Field', fontdict={'fontsize': 10, 'fontweight': 'bold', 'family': 'Arial'})
        ax1.set_xlabel('x (μm)', fontsize=9)
        ax1.set_ylabel('y (μm)', fontsize=9)
        ax1.tick_params(labelsize=8)
        ax1.axis('equal')
        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(I, cmap='plasma',
                          extent=[-self.current_params['a3']*2*1e6,
                                self.current_params['a3']*2*1e6,
                                -self.current_params['a3']*2*1e6,
                                self.current_params['a3']*2*1e6])
        ax2.set_title('Intensity',fontdict={'fontsize': 10, 'fontweight': 'bold', 'family': 'Arial'})
        ax2.set_xlabel('x (μm)', fontsize=9)
        ax2.set_ylabel('y (μm)', fontsize=9)
        ax2.tick_params(labelsize=8)
        ax2.axis('equal')
        ax3 = fig.add_subplot(133)
        phase = np.angle(E)
        im3 = ax3.imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi,
                        extent=[-self.current_params['a3']*2*1e6,
                                self.current_params['a3']*2*1e6,
                                -self.current_params['a3']*2*1e6,
                                self.current_params['a3']*2*1e6])
        ax3.set_title('Phase', fontdict={'fontsize': 10, 'fontweight': 'bold', 'family': 'Arial'})
        ax3.set_xlabel('x (μm)', fontsize=9)
        ax3.set_ylabel('y (μm)', fontsize=9)
        ax3.tick_params(labelsize=8)
        ax3.axis('equal')
        canvas = FigureCanvasTkAgg(fig, mode_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.result_plots[mode_name] = (fig, canvas)
    def update_results_table(self):
        for row in self.results_tree.get_children():
            self.results_tree.delete(row)
        for mode_name, data in self.results.items():
            self.results_tree.insert("", "end", values=(
                mode_name,
                f"{data['neff']:.6f}",
                f"{data['fval']:.2e}",
                data['iterations']
            ))
    def export_results(self):
        if not self.results:
            messagebox.showwarning("No Results", "No results to export.")
            return
        try:
            import csv
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not filename:
                return
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Mode', 'neff', 'fval', 'Iterations', 'm', 'Orientation'])
                for mode_name, data in self.results.items():
                    writer.writerow([
                        mode_name,
                        f"{data['neff']:.6f}",
                        f"{data['fval']:.2e}",
                        data['iterations'],
                        data['m'],
                        data['orientation']
                    ])
            messagebox.showinfo("Export Successful",
                              f"Results exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting results:\n{str(e)}")
    def reset_parameters(self):
        if self.calculating:
            messagebox.showwarning("Calculation in Progress",
                                 "Cannot reset during calculation.")
            return
        self.current_params = self.default_params.copy()
        for param_name, entry in self.param_entries.items():
            value = self.default_params[param_name]
            if param_name in ['a1', 'a2', 'a3', 'wavelength']:
                entry.delete(0, tk.END)
                entry.insert(0, str(value * 1e6))
            else:
                entry.delete(0, tk.END)
                entry.insert(0, str(value))
        self.update_refractive_index_plot()
        messagebox.showinfo("Parameters Reset",
                          "All parameters have been reset to default values.")
    def find_neff_interval(self, neff, space_i, parameters, start_time, timeout=30):
        k0, m, n1, n2, n3, nclad, a1, yita2, yita3 = parameters
        space = space_i / 100.0
        initial_n = neff
        continue_search = True
        neff_r = None
        neff_l = None
        while continue_search:
            serial_n = np.linspace(initial_n - space * 99, initial_n, 100)
            u1 = a1 * k0 * np.sqrt(np.abs(n1**2 - serial_n**2))
            w2 = a1 * k0 * np.sqrt(np.abs(serial_n**2 - n2**2))
            u3 = a1 * k0 * np.sqrt(np.abs(n3**2 - serial_n**2))
            w4 = a1 * k0 * np.sqrt(np.abs(serial_n**2 - nclad**2))
            with np.errstate(invalid='ignore', divide='ignore'):
                P3 = jv(m, yita2 * u3) * yv(m+1, yita3 * u3) - yv(m, yita2 * u3) * jv(m+1, yita3 * u3)
                Q3 = jv(m+1, yita2 * u3) * yv(m+1, yita3 * u3) - yv(m+1, yita2 * u3) * jv(m+1, yita3 * u3)
                R3 = jv(m, yita2 * u3) * yv(m, yita3 * u3) - yv(m, yita2 * u3) * jv(m, yita3 * u3)
                S3 = jv(m+1, yita2 * u3) * yv(m, yita3 * u3) - yv(m+1, yita2 * u3) * jv(m, yita3 * u3)
                P2b = w2 * kv(m+1, yita2 * w2) * P3 - u3 * kv(m, yita2 * w2) * Q3
                Q2b = w2 * iv(m+1, yita2 * w2) * P3 + u3 * iv(m, yita2 * w2) * Q3
                R2b = w2 * kv(m+1, yita2 * w2) * R3 - u3 * kv(m, yita2 * w2) * S3
                S2b = w2 * iv(m+1, yita2 * w2) * R3 + u3 * iv(m, yita2 * w2) * S3
                P2 = -iv(m, w2) * P2b - kv(m, w2) * Q2b
                Q2 = iv(m+1, w2) * P2b - kv(m+1, w2) * Q2b
                R2 = -iv(m, w2) * R2b - kv(m, w2) * S2b
                S2 = iv(m+1, w2) * R2b - kv(m+1, w2) * S2b
                denom_k = kv(m, yita3 * w4)
                denom_r = u1 * jv(m+1, u1) * R2 - w2 * jv(m, u1) * S2
                equation_left = np.where(denom_k != 0,
                                        w4 * kv(m+1, yita3 * w4) / denom_k,
                                        np.inf)
                equation_right = np.where(denom_r != 0,
                                         u3 * (u1 * jv(m+1, u1) * P2 - w2 * jv(m, u1) * Q2) / denom_r,
                                         np.inf)
            f_temp = equation_left - equation_right
            for i in range(98, -1, -1):
                if (not np.isnan(f_temp[i+1]) and not np.isnan(f_temp[i]) and
                    not np.isinf(f_temp[i+1]) and not np.isinf(f_temp[i])):
                    if f_temp[i+1] * f_temp[i] < 0 and abs(f_temp[i+1] * f_temp[i]) < 1.0:
                        neff_r = serial_n[i+1]
                        neff_l = serial_n[i]
                        continue_search = False
                        break
            if continue_search:
                initial_n = initial_n - 98 * space
            if time.time() - start_time > timeout:
                if neff_r is None or neff_l is None:
                    neff_r = serial_n[-1]
                    neff_l = serial_n[-2]
                continue_search = False
        space = space_i / 10.0
        return neff_l, neff_r, space
    def compute_characteristic_equation(self, neff, parameters):
        k0, m, n1, n2, n3, nclad, a1, yita2, yita3 = parameters
        u1 = a1 * k0 * np.sqrt(n1**2 - neff**2)
        w2 = a1 * k0 * np.sqrt(neff**2 - n2**2)
        u3 = a1 * k0 * np.sqrt(n3**2 - neff**2)
        w4 = a1 * k0 * np.sqrt(neff**2 - nclad**2)
        with np.errstate(invalid='ignore', divide='ignore'):
            P3 = jv(m, yita2 * u3) * yv(m+1, yita3 * u3) - yv(m, yita2 * u3) * jv(m+1, yita3 * u3)
            Q3 = jv(m+1, yita2 * u3) * yv(m+1, yita3 * u3) - yv(m+1, yita2 * u3) * jv(m+1, yita3 * u3)
            R3 = jv(m, yita2 * u3) * yv(m, yita3 * u3) - yv(m, yita2 * u3) * jv(m, yita3 * u3)
            S3 = jv(m+1, yita2 * u3) * yv(m, yita3 * u3) - yv(m+1, yita2 * u3) * jv(m, yita3 * u3)
            P2b = w2 * kv(m+1, yita2 * w2) * P3 - u3 * kv(m, yita2 * w2) * Q3
            Q2b = w2 * iv(m+1, yita2 * w2) * P3 + u3 * iv(m, yita2 * w2) * Q3
            R2b = w2 * kv(m+1, yita2 * w2) * R3 - u3 * kv(m, yita2 * w2) * S3
            S2b = w2 * iv(m+1, yita2 * w2) * R3 + u3 * iv(m, yita2 * w2) * S3
            P2 = -iv(m, w2) * P2b - kv(m, w2) * Q2b
            Q2 = iv(m+1, w2) * P2b - kv(m+1, w2) * Q2b
            R2 = -iv(m, w2) * R2b - kv(m, w2) * S2b
            S2 = iv(m+1, w2) * R2b - kv(m+1, w2) * S2b
            denom_k = kv(m, yita3 * w4)
            denom_r = u1 * jv(m+1, u1) * R2 - w2 * jv(m, u1) * S2
            if denom_k != 0 and denom_r != 0:
                equation_left = w4 * kv(m+1, yita3 * w4) / denom_k
                equation_right = u3 * (u1 * jv(m+1, u1) * P2 - w2 * jv(m, u1) * Q2) / denom_r
                fval = abs(equation_left - equation_right)
            else:
                fval = 1e-5
        return fval
    def solve_neff_for_mode(self, m, initial_neff, base_parameters, mode_name="Unknown"):
        k0, n1, n2, n3, nclad, a1, yita2, yita3 = base_parameters
        parameters = [k0, m, n1, n2, n3, nclad, a1, yita2, yita3]
        start_time = time.time()
        space = 1e-6
        fval = 1.0
        neff = initial_neff
        iterations = 0
        max_iterations = 200
        while fval > 1e-7 and iterations < max_iterations:
            iterations += 1
            neff_l, neff_r, space = self.find_neff_interval(
                neff, space, parameters, start_time
            )
            if neff_r is not None:
                neff = neff_r
            fval = self.compute_characteristic_equation(neff, parameters)
        return neff, fval, iterations
    def compute_mode_coefficients(self, neff, m, base_parameters):
        k0, n1, n2, n3, nclad, a1, yita2, yita3 = base_parameters
        u1 = a1 * k0 * np.sqrt(n1**2 - neff**2)
        w2 = a1 * k0 * np.sqrt(neff**2 - n2**2)
        u3 = a1 * k0 * np.sqrt(n3**2 - neff**2)
        w4 = a1 * k0 * np.sqrt(neff**2 - nclad**2)
        A1 = 1.0
        A2, A3, A4, A5, A6 = sp.symbols('A2 A3 A4 A5 A6')
        eq1 = jv(m+1, u1) * A1 - iv(m+1, w2) * A2 - kv(m+1, w2) * A3
        eq2 = iv(m+1, yita2 * w2) * A2 + kv(m+1, yita2 * w2) * A3 - jv(m+1, yita2 * u3) * A4 - yv(m+1, yita2 * u3) * A5
        eq3 = -iv(m, yita2 * w2) / w2 * A2 + kv(m, yita2 * w2) / w2 * A3 - jv(m, yita2 * u3) / u3 * A4 - yv(m, yita2 * u3) / u3 * A5
        eq4 = jv(m+1, yita3 * u3) * A4 + yv(m+1, yita3 * u3) * A5 - kv(m+1, yita3 * w4) * A6
        eq5 = jv(m, yita3 * u3) / u3 * A4 + yv(m, yita3 * u3) / u3 * A5 - kv(m, yita3 * w4) / w4 * A6
        try:
            solution = sp.solve([eq1, eq2, eq3, eq4, eq5], [A2, A3, A4, A5, A6])
            A2_val = float(solution[A2])
            A3_val = float(solution[A3])
            A4_val = float(solution[A4])
            A5_val = float(solution[A5])
            A6_val = float(solution[A6])
        except:
            M = np.zeros((5, 5), dtype=complex)
            b = np.zeros(5, dtype=complex)
            M[0, 0] = -iv(m+1, w2)
            M[0, 1] = -kv(m+1, w2)
            b[0] = -jv(m+1, u1) * A1
            M[1, 0] = iv(m+1, yita2 * w2)
            M[1, 1] = kv(m+1, yita2 * w2)
            M[1, 2] = -jv(m+1, yita2 * u3)
            M[1, 3] = -yv(m+1, yita2 * u3)
            M[2, 0] = -iv(m, yita2 * w2) / w2
            M[2, 1] = kv(m, yita2 * w2) / w2
            M[2, 2] = -jv(m, yita2 * u3) / u3
            M[2, 3] = -yv(m, yita2 * u3) / u3
            M[3, 2] = jv(m+1, yita3 * u3)
            M[3, 3] = yv(m+1, yita3 * u3)
            M[3, 4] = -kv(m+1, yita3 * w4)
            M[4, 2] = jv(m, yita3 * u3) / u3
            M[4, 3] = yv(m, yita3 * u3) / u3
            M[4, 4] = -kv(m, yita3 * w4) / w4
            try:
                x = np.linalg.solve(M, b)
                A2_val = x[0]
                A3_val = x[1]
                A4_val = x[2]
                A5_val = x[3]
                A6_val = x[4]
            except np.linalg.LinAlgError:
                A2_val = 0.1
                A3_val = 0.1
                A4_val = 0.1
                A5_val = 0.1
                A6_val = 0.1
        return A1, A2_val, A3_val, A4_val, A5_val, A6_val
    def compute_mode_field(self, m, orientation, neff, coefficients, base_parameters, mode_size=128):
        k0, n1, n2, n3, nclad, a1, yita2, yita3 = base_parameters
        A1, A2, A3, A4, A5, A6 = coefficients
        u1 = a1 * k0 * np.sqrt(n1**2 - neff**2)
        w2 = a1 * k0 * np.sqrt(neff**2 - n2**2)
        u3 = a1 * k0 * np.sqrt(n3**2 - neff**2)
        w4 = a1 * k0 * np.sqrt(neff**2 - nclad**2)
        x = np.linspace(-a1 * yita3 * 2, a1 * yita3 * 2, mode_size)
        y = np.linspace(-a1 * yita3 * 2, a1 * yita3 * 2, mode_size)
        X, Y = np.meshgrid(x, y)
        RHO = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y, X)
        E = np.zeros_like(RHO, dtype=complex)
        mask1 = RHO <= a1
        rho1 = RHO[mask1]
        theta1 = THETA[mask1]
        if orientation == 'even':
            E[mask1] = (A1 * jv(m, u1/a1 * rho1) / u1) * np.cos(m * theta1)
        else:
            E[mask1] = -(A1 * jv(m, u1/a1 * rho1) / u1) * np.sin(m * theta1)
        mask2 = (RHO > a1) & (RHO <= a1 * yita2)
        rho2 = RHO[mask2]
        theta2 = THETA[mask2]
        if orientation == 'even':
            E[mask2] = (-A2 * iv(m, w2/a1 * rho2) / w2 + A3 * kv(m, w2/a1 * rho2) / w2) * np.cos(m * theta2)
        else:
            E[mask2] = -(-A2 * iv(m, w2/a1 * rho2) / w2 + A3 * kv(m, w2/a1 * rho2) / w2) * np.sin(m * theta2)
        mask3 = (RHO > a1 * yita2) & (RHO <= a1 * yita3)
        rho3 = RHO[mask3]
        theta3 = THETA[mask3]
        if orientation == 'even':
            E[mask3] = (A4 * jv(m, u3/a1 * rho3) / u3 + A5 * yv(m, u3/a1 * rho3) / u3) * np.cos(m * theta3)
        else:
            E[mask3] = -(A4 * jv(m, u3/a1 * rho3) / u3 + A5 * yv(m, u3/a1 * rho3) / u3) * np.sin(m * theta3)
        mask4 = RHO > a1 * yita3
        rho4 = RHO[mask4]
        theta4 = THETA[mask4]
        if orientation == 'even':
            E[mask4] = (A6 * kv(m, w4/a1 * rho4) / w4) * np.cos(m * theta4)
        else:
            E[mask4] = -(A6 * kv(m, w4/a1 * rho4) / w4) * np.sin(m * theta4)
        I = np.abs(E)**2
        E_max = np.max(np.abs(E))
        if E_max > 0:
            E = E / E_max
            I = I / np.max(I)
        return E, I
def main():
    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
        print("Please use Python 3.6 or higher")
        sys.exit(1)
    try:
        import numpy as np
        from scipy.special import jv, yv, iv, kv
        import sympy as sp
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install numpy matplotlib scipy sympy")
        sys.exit(1)
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    try:
        root = tk.Tk()
        app = FiberModeSolverGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure tkinter is installed:")
        print("   - Ubuntu/Debian: sudo apt-get install python3-tk")
        print("   - macOS: Usually pre-installed")
        print("   - Windows: Usually pre-installed")
        print("2. Try running in a clean Python environment")
        sys.exit(1)
if __name__ == "__main__":
    main()
