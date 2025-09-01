import matplotlib.pyplot as plt
import numpy as np

def plot_time_series(data, title='Time Series', xlabel='Time', ylabel='Value'):
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Data', color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def plot_histogram(data, bins=30, title='Histogram', xlabel='Value', ylabel='Frequency'):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, color='orange', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def plot_scatter(x, y, title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis'):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color='green', alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def plot_heatmap(data, title='Heatmap', xlabel='X-axis', ylabel='Y-axis'):
    plt.figure(figsize=(8, 6))
    plt.imshow(data, aspect='auto', cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label='Intensity')
    plt.show()