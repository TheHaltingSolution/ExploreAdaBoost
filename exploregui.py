import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)

from collections import namedtuple
GuiElements = namedtuple('GuiElements', ['iteration_label', 'true_cost_label', 'training_cost_label',
                                         'generalization_error_label'])
PlotElements = namedtuple('PlotElements', ['figure', 'subplot', 'plot1_line_y', 'plot2_vertical_marker'])


class ExploreGui:
    def __init__(self):
        self.window = tk.Tk()

    def setup_ui(self, model_outputs):
        greeting = tk.Label(text="Iteration")
        greeting.pack()
        iteration_label = tk.Label(self.window)
        iteration_label.pack()
        iteration_var = tk.IntVar()

        true_cost_label = tk.Label(self.window, fg='black')
        true_cost_label.pack()
        training_cost_label = tk.Label(self.window, fg='green')
        training_cost_label.pack()
        generalization_error_label = tk.Label(self.window, fg='blue')
        generalization_error_label.pack()

        gui_elements = GuiElements(iteration_label, true_cost_label, training_cost_label, generalization_error_label)
        plot_elements = self.plot_prepare(model_outputs)

        btn_increase = tk.Button(master=self.window, text="+", width=5, height=1,
                                 command=lambda x=None: self.increase(iteration_var, gui_elements, plot_elements, model_outputs))

        btn_decrease = tk.Button(master=self.window, text="-", width=5, height=1,
                                 command=lambda x=None: self.decrease(iteration_var, gui_elements, plot_elements, model_outputs))

        number_of_iterations = len(model_outputs.prediction_functions) - 1
        scale = tk.Scale(self.window, variable=iteration_var, from_=0, to_=number_of_iterations,
                         command=lambda x=None: self.update_iteration(iteration_var, gui_elements, plot_elements, model_outputs),
                         orient=tk.HORIZONTAL, length=1000)

        btn_decrease.pack(ipadx=10, ipady=10, expand=True, fill='both', side='left')
        btn_increase.pack(ipadx=10, ipady=10, expand=True, fill='both', side='left')
        scale.pack(anchor=tk.CENTER)

    def start_ui(self):
        self.window.mainloop()

    def increase(self, iteration_var, gui_elements, plot_elements, model_outputs):
        iteration_var.set(iteration_var.get() + 1)
        self.update_iteration(iteration_var, gui_elements, plot_elements, model_outputs)

    def decrease(self, iteration_var, gui_elements, plot_elements, model_outputs):
        iteration_var.set(iteration_var.get() - 1)
        self.update_iteration(iteration_var, gui_elements, plot_elements, model_outputs)

    def update_iteration(self, iteration_var, gui_elements, plot_elements, model_outputs):
        iteration = iteration_var.get()
        selection = "Value = " + str(iteration)
        gui_elements.iteration_label.config(text=selection)
        gui_elements.true_cost_label.config(text="True Cost = " + str(model_outputs.true_error[iteration]))
        gui_elements.training_cost_label.config(text="Training Cost = " + str(model_outputs.training_error[iteration]))
        gui_elements.generalization_error_label.config(text="Generalization Error = " + str(model_outputs.generalization_error[iteration]))
        self.plot_update(iteration, plot_elements, model_outputs.prediction_functions[iteration])

    def plot_update(self, iteration, plot_elements, prediction_function):
        plot_elements.plot1_line_y.set_ydata(prediction_function)
        plot_elements.plot2_vertical_marker.set_xdata(iteration)
        plot_elements.figure.canvas.draw()
        plot_elements.figure.canvas.flush_events()


    def plot_prepare(self, model_outputs):
        # the figure that will contain the plot
        fig = Figure(figsize=(14, 5),
                     dpi=100)

        X_true_one_dim = model_outputs.sample.X_true[:, 0]
        X_sample_one_dim = model_outputs.sample.X_sample[:, 0]

        # adding the subplot
        plot1 = fig.add_subplot(121)
        plot2 = fig.add_subplot(122)

        # plot x-axis, visualize threshold
        x_axis = [0 for i in X_true_one_dim]
        plot1.plot(X_true_one_dim, x_axis, 'k-')

        # add training set
        plot1.plot(X_sample_one_dim, model_outputs.sample.y_sample, 'k.', markersize=1)

        # Plot true model
        plot1.plot(X_true_one_dim, model_outputs.sample.y_true, 'k-')

        # plotting the prediction function
        line_y, = plot1.plot(X_true_one_dim, x_axis, 'g-')
        plot1.set_ylim(model_outputs.axis_range.min, model_outputs.axis_range.max)

        # prepare plot 2
        iterations = range(len(model_outputs.true_error))
        plot2.plot(iterations, model_outputs.true_error, 'k-')
        plot2.plot(iterations, model_outputs.training_error, 'g-')
        plot2.plot(iterations, model_outputs.generalization_error, 'b-')
        vertical_marker = plot2.axvline(x=0, color='r')

        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg(fig,
                                   master=self.window)
        canvas.draw()

        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().pack()

        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas,
                                       self.window)
        toolbar.update()

        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack()
        return PlotElements(fig, plot1, line_y, vertical_marker)
