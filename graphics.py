import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

def correlogram(matrix=None, dec=1, title='Correlogram', valmin=-1, valmax=1):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(matrix, dec), vmin=valmin, vmax=valmax, cmap='bwr', annot=True)

def intensity_map(matrix=None, dec=1, title='Intensity Map', valmin=None, valmax=None):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(matrix, dec), vmin=valmin, vmax=valmax, cmap='Reds', annot=True)

def principalComponents(eigenvalues=None, XLabel='Principal components', YLabel='Eigenvalues (variance)',
                        title='Explained variance by the principal components'):
    plt.figure(title, figsize=(13, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(XLabel, fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(YLabel, fontsize=14, color='k', verticalalignment='bottom')
    components = ['C'+str(j+1) for j in range(eigenvalues.shape[0])]
    plt.plot(components, eigenvalues, 'bo-')
    plt.axhline(y=1, color='r')

def show():
    plt.show()
