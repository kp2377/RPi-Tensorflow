from tkinter import *
import tkinter as tk

from email.policy import default
from tkinter import ttk
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import os
import cv2
import numpy as np

import argparse
import sys
import csv
import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DATE = 2
HOUR = 3
MINUTES = 4
SECONDS = 5
NUMBER = 6

def read_csv(file_name, rights):
    with open(file_name, rights) as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        del data[0]
        return data;

minutes_group_data = []
minutes_group_names = []

hours_group_data = []
hours_group_names = []

day_group_data = []
day_group_names = []
   

    

    



LARGE_FONT = ("Verdana", 12)


class MakerSpacePeopleCounter(tk.Tk):
    def __init__(self,master = None, *args, **kwargs):
        tk.Tk.__init__(self,master, *args, **kwargs)


        self.master = master
        menu = Menu(self.master)
        tk.Tk.config(self, menu=menu)
        file = Menu(menu)
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)
        edit = Menu(menu)
        edit.add_command(label="Undo")
        edit.add_command(label="Show Img", command=lambda: self.show_frame(week_graph))
        menu.add_cascade(label="Edit", menu=edit)


        tk.Tk.iconbitmap(self)
        tk.Tk.wm_title(self, "MakerSpaceCounter")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)



        self.frames = {}

        for F in (main_window, day_graph, week_graph, month_graph):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(main_window)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def showImg(self):
        load = Image.open("kartikey.png")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)

    def showText(self):
        text = Label(self, text="Hey there good lookin!")
        text.pack()

    def client_exit(self):
        exit()




class main_window(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Main Option Window", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button = ttk.Button(self, text="Visit Minutes Graph", command=lambda: controller.show_frame(day_graph))
        button.pack()

        button2 = ttk.Button(self, text="Visit Hour Graph",command=lambda: controller.show_frame(week_graph))
        button2.pack()

        button3 = ttk.Button(self, text="Visit Day Graph",command=lambda: controller.show_frame(month_graph))
        button3.pack()

class day_graph(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="This represents number of persons in a minute!!!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame(main_window))
        button1.pack()
        with open("minute_logs.csv", "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]
            del data[0]
            rows = len(data)  #this is the way you decide the number of rows in a list
            total_entries = rows
            #columns = len(data[0]) # this is the way to determine the nmumber of columns in the list
            for i in range(rows):
                minutes_group_data.append(int(data[i][NUMBER]))
                minutes_group_names.append(data[i][SECONDS])
                minutes_group_mean = np.mean(minutes_group_data)
        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.barh(minutes_group_names, minutes_group_data)

        canvas = FigureCanvasTkAgg(f, self)
        #canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class week_graph(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Number of persons in an hour", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame(main_window))
        button1.pack()
        
        with open("hour_logs.csv", "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]
            del data[0]
            rows = len(data)  #this is the way you decide the number of rows in a list
            total_entries = rows
            #columns = len(data[0]) # this is the way to determine the nmumber of columns in the list
            for i in range(rows):
                hours_group_data.append(int(data[i][NUMBER]))
                hours_group_names.append(data[i][MINUTES])
            hours_group_mean = np.mean(hours_group_data)        
        
        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.barh(hours_group_names, hours_group_data)

        canvas = FigureCanvasTkAgg(f, self)
        #canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


class month_graph(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Number of persons in a day", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame(main_window))
        button1.pack()
        
        
        with open("day_logs.csv", "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]
            del data[0]
            rows = len(data)  #this is the way you decide the number of rows in a list
            total_entries = rows
            #columns = len(data[0]) # this is the way to determine the nmumber of columns in the list
            for i in range(rows):
                day_group_data.append(int(data[i][NUMBER]))
                day_group_names.append(data[i][HOUR])
            day_group_mean = np.mean(day_group_data)

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.barh(day_group_names, day_group_data)

        canvas = FigureCanvasTkAgg(f, self)
        #canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


app = MakerSpacePeopleCounter()
app.after(1000)
app.mainloop()
