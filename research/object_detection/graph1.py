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

#If you have to accesss each cell
with open("person_log.csv", "r") as f:
    reader = csv.reader(f)
    data = [row for row in reader]
    del data[0]
    rows = len(data)  #this is the way you decide the number of rows in a list
    total_entries = rows
    columns = len(data[0]) # this is the way to determine the nmumber of columns in the list
    add = 0
    for i in range(rows):
        print("The sum before add is :", add)
        add = add + int(data[i][6])
        if((int(data[i][6])) == 0):
            total_entries = total_entries - 1
    #del data[0]
    print("The final addition is:", add)
    print("The total number of entries are :", total_entries)
    if(total_entries):
        average = add/total_entries
    else:
        average = 0
    print(int(average))
    
#if you have to access the rows details only
#with open("person_log.csv", "r") as f:
#    reader = csv.reader(f)
#    for row in reader:
#        #print(row[5])
#        listen = list(row[5])
#        print(listen)

