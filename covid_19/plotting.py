import matplotlib.pyplot as plt 
from matplotlib import rcParams 
import numpy as np

def latexconfig():
    """Paper ready plots; standard LaTeX configuration."""
    pgf_latex = {                                       # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",                    # change this if using xetex or lautex
        "text.usetex": True,                            # use LaTeX to write all text
        "font.family": "serif",                         # default LaTex option
        "font.serif": [],                               # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,                           # LaTeX default is 10pt font
        "font.size": 10,                                # LaTeX default is 10pt font
        "legend.fontsize": 10,                          # Make the legend/label fonts a little smaller
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": figsize(1.0),                 # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",            # utf8 input support
            r"\usepackage[T1]{fontenc}",                # plots will be generated using this preamble
            ]
        }
    rcParams.update(pgf_latex)

def figsize(scale, nplots=1):
    """Golden ratio between the width and height: the ratio 
    is the same as the ratio of their sum to the width of 
    the figure. 
    
    width + height    height
    -------------- = --------
         width        width

    Props for the code goes to: https://github.com/maziarraissi/PINNs/blob/master/Utilities/plotting.py
    """
    fig_width_pt = 390.0                               # LaTeX \the\textwidth
    inches_per_pt = 1.0/72.27                          # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0               # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt*scale       # width in inches
    fig_height = fig_width*golden_mean*nplots          # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def plotData(x, data, log=False, *args):
    """Plot the data in log or linear scale."""     
    if isinstance(data, np.ndarray):
        confirmed = data
    else: raise ValueError('np.ndarray data type is required.')

    if args:
        if len(args) == 1:
            scale = args[0]
            nplots = 1
        elif len(args) == 2:
            scale, nplots = args
        else: raise ValueError('Too many items to unpack.')
    else:
        scale = 1
        nplots = 1
    
    if not log:
        l = ''
        fig = plt.figure(figsize=figsize(scale, nplots))
        ax = fig.add_subplot(111)
        ax.plot(confirmed, marker='x', markersize=9,
                 linestyle='None', color='blue', 
                 label='confirmed cases')
        
    else:
        l = 'log'
        fig = plt.figure(figsize=figsize(scale, nplots))
        ax = fig.add_subplot(111)
        ax.plot(np.log10(confirmed), marker='x', markersize=9,
                 linestyle='None', color='blue', 
                 label='log of the number of confirmed cases')

    ax.set_xlabel('day')
    ax.set_ylabel('{} the number of confirmed cases'.format(l))
    
    return fig, ax

# short test
if __name__ == "__main__":
    latexconfig()   
    x = np.linspace(0, 10, 10)
    data = np.sin(2*x + 1/2)
    plotData(x, data)
    plt.legend()
    plt.show()
