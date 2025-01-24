#Baraa Nasar 1210880

import random  # Importing the random module for generating random numbers
from typing import List  # Importing List from typing module for type hinting
import tkinter as tk  # Importing tkinter for GUI
from tkinter import ttk, messagebox  # Importing ttk for themed widgets and messagebox for dialog boxes
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import matplotlib.dates as mdates  # Importing matplotlib.dates for handling date formatting in plots
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Importing FigureCanvasTkAgg to embed matplotlib plots in tkinter
from datetime import datetime, timedelta  # Importing datetime and timedelta for handling dates and times

# Class to represent an operation in a job
class Operation:
    def __init__(self, machine_id: str, processing_time: int):
        self.machine_id = machine_id  # ID of the machine
        self.processing_time = processing_time  # Processing time for the operation

# Class to represent a job with multiple operations
class Job:
    def __init__(self, job_id: str, operations: List[Operation]):
        self.job_id = job_id  # ID of the job
        self.operations = operations  # List of operations for the job

# Class to represent a schedule with jobs and machines
class Schedule:
    def __init__(self, jobs: List[Job], machines: List[str]):
        self.jobs = jobs  # List of jobs
        self.machines = machines  # List of machines
        self.machine_schedules = {machine: [] for machine in machines}  # Dictionary to store schedules for each machine
        self.fitness = float('inf')  # Fitness value of the schedule

    # Method to randomize the operations within jobs
    def randomize(self):
        for job in self.jobs:
            random.shuffle(job.operations)  # Shuffle operations within each job
        self.calculate_fitness()  # Calculate fitness of the randomized schedule

    # Method to calculate the fitness of the schedule
    def calculate_fitness(self):
        self.machine_schedules = {machine: [] for machine in self.machines}  # Reset machine schedules
        time = {machine: 0 for machine in self.machines}  # Dictionary to keep track of time for each machine
        for job in self.jobs:
            current_time = 0  # Initialize current time
            for operation in job.operations:
                machine = operation.machine_id  # Get the machine ID for the operation
                start_time = max(current_time, time[machine])  # Calculate the start time
                end_time = start_time + operation.processing_time  # Calculate the end time
                self.machine_schedules[machine].append((job.job_id, start_time, end_time))  # Append the operation to the machine schedule
                time[machine] = end_time  # Update the time for the machine
                current_time = end_time  # Update the current time
        self.fitness = max(time.values())  # Set the fitness value as the maximum end time

    # Method to perform crossover between two schedules
    def crossover(self, other):
        child = Schedule(self.jobs, self.machines)  # Create a new child schedule
        for job in self.jobs:
            split_point = random.randint(1, len(job.operations) - 1)  # Determine a split point for crossover
            new_ops = job.operations[:split_point] + other.jobs[self.jobs.index(job)].operations[split_point:]  # Combine operations from both parents
            child.jobs[self.jobs.index(job)].operations = new_ops  # Set the operations for the child job
        child.calculate_fitness()  # Calculate the fitness of the child schedule
        return child  # Return the child schedule

    # Method to mutate the schedule
    def mutate(self):
        for job in self.jobs:
            if random.random() < 0.1:  # With a probability of 0.1
                random.shuffle(job.operations)  # Shuffle operations within the job
        self.calculate_fitness()  # Calculate fitness of the mutated schedule

# Function to run the genetic algorithm for job scheduling
def genetic_algorithm(jobs, machines, generations, population_size):
    try:
        population = [Schedule(jobs, machines) for _ in range(population_size)]  # Initialize population with schedules
        for schedule in population:
            schedule.randomize()  # Randomize each schedule
        for generation in range(generations):
            population.sort(key=lambda x: x.fitness)  # Sort population based on fitness
            new_population = population[:population_size // 2]  # Select the top half of the population
            for _ in range(population_size // 2, population_size):
                parent1 = random.choice(new_population)  # Select a random parent from the top half
                parent2 = random.choice(new_population)  # Select another random
                child = parent1.crossover(parent2)  # Create a child schedule by crossover
                child.mutate()  # Mutate the child schedule
                new_population.append(child)  # Add the child to the new population
            population = new_population  # Update the population with the new generation
        population.sort(key=lambda x: x.fitness)  # Sort the final population
        return population[0]  # Return the best schedule (schedule with the lowest fitness)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during genetic algorithm execution: {e}")  # Display error message if an exception occurs

# Class representing the application for Job Shop Scheduling
class JobShopSchedulerApp:
    def __init__(self, root):
        self.root = root  # Reference to the root window
        self.root.title("Job Shop Scheduler")  # Set the title of the window
        self.root.state('zoomed')  # Maximize the window
        self.jobs = []  # Initialize list to store jobs
        self.machines = []  # Initialize list to store machines
        self.best_schedule = None  # Initialize variable to store the best schedule
        self.create_widgets()  # Create GUI widgets

    # Method to create GUI widgets
    def create_widgets(self):
        self.input_frame = ttk.Frame(self.root)  # Create a frame for input widgets
        self.input_frame.pack(pady=10, fill='x', expand=True)  # Pack the input frame

        ttk.Label(self.input_frame, text="Machine IDs (comma separated):").grid(row=0, column=0)  # Label for machine IDs
        self.machine_entry = ttk.Entry(self.input_frame)  # Entry widget for machine IDs
        self.machine_entry.grid(row=0, column=1)  # Grid placement for machine entry

        ttk.Label(self.input_frame, text="Jobs (e.g., Job_1:M1[10]->M2[5]->M4[12], Job_2:M2[7]->M3[15]->M1[8]):").grid(row=1, column=0)  # Label for job descriptions
        self.jobs_entry = ttk.Entry(self.input_frame, width=50)  # Entry widget for job descriptions
        self.jobs_entry.grid(row=1, column=1)  # Grid placement for job entry

        self.run_button = ttk.Button(self.input_frame, text="Run Scheduler", command=self.run_scheduler)  # Button to run the scheduler
        self.run_button.grid(row=2, column=0, columnspan=2)  # Grid placement for run button

        self.gantt_button = ttk.Button(self.input_frame, text="Show Gantt Chart", command=self.show_gantt_chart)  # Button to show Gantt chart
        self.gantt_button.grid(row=3, column=0, columnspan=2)  # Grid placement for Gantt chart button

        self.table_button = ttk.Button(self.input_frame, text="Show Table Chart", command=self.show_table_chart)  # Button to show table chart
        self.table_button.grid(row=4, column=0, columnspan=2)  # Grid placement for table chart button

        self.output_frame = ttk.Frame(self.root)  # Create a frame for output widgets
        self.output_frame.pack(pady=10, fill='both', expand=True)  # Pack the output frame

    # Method to parse job descriptions entered by the user
    def parse_jobs(self, jobs_str: str) -> List[Job]:
        try:
            jobs = []  # Initialize list to store parsed jobs
            for job_str in jobs_str.split(','):  # Split job descriptions separated by comma
                job_id, operations_str = job_str.split(':')  # Split job ID and operations
                operations = []  # Initialize list to store job operations
                for op_str in operations_str.split('->'):  # Split operations separated by '->'
                    machine_id, processing_time = op_str.split('[')  # Split machine ID and processing time
                    processing_time = int(processing_time.strip().strip(']'))  # Convert processing time to integer
                    operations.append(Operation(machine_id.strip(), processing_time))  # Create Operation object and add to operations list
                jobs.append(Job(job_id.strip(), operations))  # Create Job object and add to jobs list
            return jobs  # Return the list of parsed jobs
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse jobs: {e}")  # Display error message if parsing fails
            return []  # Return an empty list

    # Method to run the scheduler
    def run_scheduler(self):
        try:
            machine_ids = self.machine_entry.get().split(',')  # Get machine IDs entered by the user
            if not machine_ids or not self.jobs_entry.get():  # Check if machine IDs or job descriptions are empty
                messagebox.showerror("Error", "Please enter machine IDs and jobs.")
                return  # Display error message and return if machine IDs or job descriptions are empty

            self.machines = [m.strip() for m in machine_ids]  # Strip whitespace from machine IDs and store in a list
            self.jobs = self.parse_jobs(self.jobs_entry.get())  # Parse job descriptions entered by the user

            self.best_schedule = genetic_algorithm(self.jobs, self.machines, generations=100, population_size=50)  # Run genetic algorithm to find the best schedule
            if self.best_schedule:  # If a best schedule is found
                messagebox.showinfo("Success", "Scheduling completed successfully!")  # Display success message
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while running the scheduler: {e}")  # Display error message if an exception occurs

    # Method to show the Gantt chart
    def show_gantt_chart(self):
        if not self.best_schedule:  # If no best schedule is found
            messagebox.showerror("Error", "Please run the scheduler first.")  # Display error message
            return
        self.display_gantt_chart(self.best_schedule)  # Display Gantt chart for the best schedule

    # Method to show the table chart
    def show_table_chart(self):
        if not self.best_schedule:  # If no best schedule is found
            messagebox.showerror("Error", "Please run the scheduler first.")  # Display error message
            return
        self.display_table_chart(self.best_schedule)  # Display table chart for the best schedule

    # Method to display the Gantt chart
    def display_gantt_chart(self, schedule: Schedule):
        try:
            for widget in self.output_frame.winfo_children():
                widget.destroy()  # Clear previous widgets from the output frame

            fig, ax = plt.subplots(figsize=(20, 15))  # Create a figure and axis for the Gantt chart

            cmap = plt.get_cmap('tab20')  # Get a colormap for coloring different jobs
            colors = [cmap(i) for i in range(len(schedule.jobs))]  # Generate colors for each job

            current_datetime = datetime.now()  # Get the current date and time

            job_details = []  # Initialize list to store job details

            for machine in schedule.machine_schedules:
                schedule.machine_schedules[machine].sort(key=lambda x: x[1])  # Sort operations based on start time

            row_counter = 0  # Initialize row counter for plotting
            machine_to_rows = {machine: [] for machine in schedule.machines}  # Dictionary to map machines to rows

            for machine, operations in schedule.machine_schedules.items():
                for job_id, start_time, end_time in operations:
                    start = current_datetime + timedelta(minutes=start_time)  # Calculate start time
                    end = current_datetime + timedelta(minutes=end_time)  # Calculate end time
                    duration = end - start  # Calculate duration
                    job_index = schedule.jobs.index(next(job for job in schedule.jobs if job.job_id == job_id))  # Get index of the job
                    color = colors[job_index]  # Get color for the job

                    row_name = f"{machine} - Job {job_id}"  # Generate row name
                    ax.barh(row_name, duration, left=start, color=color, edgecolor='black', height=0.4, align='center')  # Plot the bar for the operation

                    job_details.append(f"{job_id} on {machine}: Start: {start.strftime('%H:%M:%S')} End: {end.strftime('%H:%M:%S')}")  # Add job details

                    ax.axvline(start, color='black', linestyle='--', linewidth=0.8)  # Add vertical line for start time
                    ax.text(start, row_counter, start.strftime('%H:%M:%S'), rotation=90, verticalalignment='bottom', fontsize=8)  # Add text for start time
                    ax.axvline(end, color='red', linestyle='--', linewidth=0.8)  # Add vertical line for end time
                    ax.text(end, row_counter, end.strftime('%H:%M:%S'), rotation=90, verticalalignment='bottom', fontsize=8)  # Add text for end time

                    machine_to_rows[machine].append(row_name)  # Map machine to row
                    row_counter += 1  # Increment row counter

            ax.xaxis_date()  # Set x-axis as dates
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))  # Set major locator for x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Set date formatter for x-axis
            plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis ticks
            plt.yticks(fontsize=10)  # Set font size for y-axis ticks
            plt.xlabel('Time', fontsize=10)  # Set label for x-axis
            plt.ylabel('Machines and Jobs', fontsize=10)  # Set label for y-axis
            plt.suptitle('Gantt Chart', fontsize=12, x=0.05, ha='left')  # Set title for the plot

            handles = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(len(schedule.jobs))]  # Create handles for legend
            labels = [job.job_id for job in schedule.jobs]  # Create labels for legend
            plt.legend(handles=handles, labels=labels, fontsize=10)  # Add legend to the plot

            canvas = FigureCanvasTkAgg(fig, master=self.output_frame)  # Create canvas for embedding the plot
            canvas.draw()  # Draw the plot on the canvas
            canvas.get_tk_widget().pack()  # Pack the canvas into the output frame

            self.display_job_details(job_details)  # Display job details

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while displaying the Gantt chart: {e}")  # Display error message if an exception occurs

    # Method to display the table chart
    def display_table_chart(self, schedule: Schedule):
        try:
            for widget in self.output_frame.winfo_children():
                widget.destroy()  # Clear previous widgets from the output frame

            fig, ax = plt.subplots(figsize=(20, len(schedule.jobs) + 5))  # Create a figure and axis for the table chart

            min_time = float('inf')  # Initialize minimum time
            max_time = 0  # Initialize maximum time
            for machine, operations in schedule.machine_schedules.items():
                for _, start_time, end_time in operations:
                    if start_time < min_time:
                        min_time = start_time  # Update minimum time
                    if end_time > max_time:
                        max_time = end_time  # Update maximum time

            interval_minutes = 10  # Set interval for time intervals
            time_intervals = list(range(min_time, max_time + interval_minutes, interval_minutes))  # Create list of time intervals
            headers = ["Job/Machine"] + [str(t) for t in time_intervals]  # Create headers for the table chart

            cell_text = []  # Initialize list to store cell text
            cell_colors = []  # Initialize list to store cell colors

            cmap = plt.get_cmap('tab20')  # Get a colormap for coloring different jobs
            colors = {job.job_id: cmap(i % 20) for i, job in enumerate(schedule.jobs)}  # Generate colors for each job

            for machine, operations in schedule.machine_schedules.items():
                for job_id, start_time, end_time in operations:
                    row_text = [f"{machine} - {job_id}"]  # Create row text with machine and job ID
                    row_colors = ['white']  # Initialize row colors

                    for t in time_intervals:
                        if start_time <= t < end_time:
                            row_text.append(" ")  # Add empty string for the time interval
                            row_colors.append(colors[job_id])  # Add color corresponding to job ID
                        else:
                            row_text.append(" ")  # Add empty string for the time interval
                            row_colors.append('white')  # Set color as white for non-active time intervals

                    cell_text.append(row_text)  # Append row text to cell text
                    cell_colors.append(row_colors)  # Append row colors to cell colors

            table = ax.table(cellText=cell_text, cellLoc='center', loc='center', colLabels=headers, cellColours=cell_colors)  # Create table with cell text and colors

            table.auto_set_font_size(False)  # Disable automatic font size adjustment
            table.set_fontsize(10)  # Set font size for the table
            table.scale(1.2, 1.2)  # Scale the table size

            for key, cell in table.get_celld().items():
                cell.set_edgecolor('grey')  # Set edge color for cells
                if key[0] == 0:
                    cell.set_text_props(weight='bold')  # Set text properties for header cells
                    cell.set_facecolor('lightgrey')  # Set face color for header cells

            ax.xaxis.set_visible(False)  # Hide x-axis
            ax.yaxis.set_visible(False)  # Hide y-axis
            ax.set_frame_on(False)  # Turn off frame

            canvas = FigureCanvasTkAgg(fig, master=self.output_frame)  # Create canvas for embedding the plot
            canvas.draw()  # Draw the plot on the canvas
            canvas.get_tk_widget().pack()  # Pack the canvas into the output frame

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while displaying the table chart: {e}")  # Display error message if an exception occurs

    # Method to display job details
    def display_job_details(self, job_details):
        for widget in self.input_frame.winfo_children():
            if isinstance(widget, tk.Label) and widget.grid_info()["row"] > 1:
                widget.destroy()  # Clear previous job details labels from the input frame

        row = 3  # Initialize row counter
        for detail in job_details:
            label = tk.Label(self.input_frame, text=detail, anchor='w')  # Create label for job detail
            label.grid(row=row, column=0, columnspan=2, sticky='w')  # Grid placement for label
            row += 1  # Increment row counter

if __name__ == "__main__":
    root = tk.Tk()  # Create the root window
    app = JobShopSchedulerApp(root)  # Create an instance of the application
    root.mainloop()  # Run the application

