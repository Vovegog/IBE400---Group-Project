import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay, r2_score, mean_squared_error



def histplot_maker(x):
    """
    Generates a grid of histogram plots for each column in the given DataFrame.
    Parameters:
    x (pd.DataFrame): The input DataFrame containing the data to be plotted.
    The function calculates the number of rows and columns needed to display
    all histograms in a grid layout. Each histogram is plotted using seaborn's
    histplot function with a kernel density estimate (KDE) overlay.
    Returns:
    None: The function displays the plots and does not return any value.
    """
    n = len(x.columns)
    plotcols = 5
    if n % plotcols == 0:
        plotrows = n // plotcols
    else:
        plotrows = (n // plotcols) + 1

    fig, axs = plt.subplots(plotrows, plotcols, figsize=(18, 2 * plotrows))
    axs = axs.ravel()

    for i in range(n):
        sns.histplot(x=x.columns[i], data=x, ax=axs[i], kde=True)
    
    plt.tight_layout()
    plt.show()

# Boxplots
def boxplot_maker(x):
    """
    Generates a grid of boxplots for each column in the given DataFrame.
    Parameters:
    x (pd.DataFrame): The input DataFrame containing the data to be plotted.
    The function calculates the number of rows and columns needed to display 
    all boxplots in a grid format. It then creates the boxplots using seaborn 
    and displays them using matplotlib.
    Example:
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [2, 3, 4, 5, 6]
    ... })
    >>> boxplot_maker(data)
    """
    n = len(x.columns)
    plotcols = 5
    if n % plotcols == 0:
        plotrows = n // plotcols
    else:
        plotrows = (n // plotcols) + 1

    fig, axs = plt.subplots(plotrows, plotcols, figsize=(18, 2 * plotrows))
    axs = axs.ravel()

    for i in range(n):
        sns.boxplot(y=x.columns[i], data=x, ax=axs[i])
    
    plt.tight_layout()
    plt.show()

# Violinplot
def violinplot_maker(x):
    """
    Generates a grid of violin plots for each column in the given DataFrame.
    Parameters:
    x (pd.DataFrame): The input DataFrame containing the data to be plotted. Each column in the DataFrame will have its own violin plot.
    Returns:
    None: The function displays the violin plots using matplotlib and seaborn.
    Notes:
    - The function calculates the number of rows and columns needed for the subplot grid based on the number of columns in the DataFrame.
    - The subplot grid will have a maximum of 5 columns.
    - The function uses seaborn's violinplot to create each individual plot.
    - The layout is adjusted to be tight to avoid overlapping plots.
    """
    n = len(x.columns)
    plotcols = 5
    if n % plotcols == 0:
        plotrows = n // plotcols
    else:
        plotrows = (n // plotcols) + 1

    fig, axs = plt.subplots(plotrows, plotcols, figsize=(18, 2 * plotrows))
    axs = axs.ravel()

    for i in range(n):
        sns.violinplot(y=x.columns[i], data=x, ax=axs[i])
    
    plt.tight_layout()
    plt.show()

# Barplot maker
def barplot_maker(x, y, data):
    """
    Creates a series of bar plots for each column in the provided DataFrame.
    Parameters:
    x (pd.DataFrame): DataFrame containing the columns to be plotted on the y-axis.
    y (str): Column name to be plotted on the x-axis.
    data (pd.DataFrame): DataFrame containing the data to be plotted.
    Returns:
    None: This function displays the bar plots and does not return any value.
    """
    n = len(x.columns)
    plotcols = 5
    if n % plotcols == 0:
        plotrows = n // plotcols
    else:
        plotrows = (n // plotcols) + 1

    fig, axs = plt.subplots(plotrows, plotcols, figsize=(18, 3 * plotrows))
    axs = axs.ravel()

    for i in range(n):
        sns.barplot(y=x.columns[i], x=y, data=data, ax=axs[i])
    
    plt.tight_layout()
    plt.show()

def plotCluster(data, kMeans):
    """
    Plots a scatter plot matrix of the clusters formed by k-means clustering.
    Parameters:
    data (pandas.DataFrame): The input data containing the features to be plotted.
    kMeans (sklearn.cluster.KMeans): The fitted k-means clustering model.
    Returns:
    None
    """
    n = len(data.columns)
    figLength = n * 6

    plt.figure(figsize=(25, figLength))

    for i, (x, y) in enumerate(itertools.combinations(data.columns, 2)):
        plt.subplot(n, 3, i + 1)
        scatter = plt.scatter(data[x], data[y], c=kMeans.labels_, cmap='tab10')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend(*scatter.legend_elements(), title="Clusters")

    plt.show()

def r2MSETest(y_test, y_pred):
    """
    Calculate and print the Mean Squared Error (MSE) and R-squared (R2) score for the given test and predicted values.

    Parameters:
    y_test (array-like): True values of the target variable.
    y_pred (array-like): Predicted values of the target variable.

    Returns:
    None
    """
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse} \nR-squared: {r2}')
