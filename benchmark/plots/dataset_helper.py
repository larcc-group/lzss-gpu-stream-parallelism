import sqlite3
import numpy as np


import math
class StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        return math.sqrt(self.S / (self.k-2))

class DatasetHelper:
    def __init__(self):
        self.database_name = "../benchmark.db"
    
    def __enter__(self):
        self.db = sqlite3.connect(self.database_name)
        self.db.create_aggregate("stdev", 1, StdevFunc)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.db.close()

    def select_data(self,sql):
        cursor = self.db.cursor()
        cursor.execute(sql)
        return cursor
    
    def select_vector(self,sql):
        cursor = self.db.cursor()
        cursor.execute(sql)

        return  np.array([x[0] for x in cursor])

def better_name(filename):
    return filename.split("/")[-1].split(".")[0]

current_colors = {}
available_colors = [
    ("blue","s","\\\\"),
    ("green","*","//"),
    ("red",">","\\"),
    ("cyan","8","//"),
    ("magenta","o"),
    ("yellow","p"),
    ("black","h"),
    ("white","<")
]
def color_for(name):
    if name not in current_colors:
        current_colors[name] = available_colors.pop(0)
    return current_colors[name]