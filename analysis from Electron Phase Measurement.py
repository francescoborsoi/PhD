#this contains functions that were before in Insect/Hashtag Analysis.ipynb
#script written by Francesco on 02/04/2017 to analyze transport measurements
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
import math
import cmath
import scipy as sci
from scipy.signal import savgol_filter
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import scipy.special as sp
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.collections as collections
import importlib
from IPython.display import Math
from scipy import optimize
from scipy.signal import find_peaks_cwt
from matplotlib.colors import LogNorm
import time

title_font=14
axes_font=14
tick_font=14
marker_size=6
text_font=16
caption_font=18
legend_font = 10

sub_hspace=8
sub_wspace=1

limit_fit = 800
phi_0 = 2.*2.0678e-15


#contact resistance
R_c = 8000

# functions definition

A2 = 0
B2 = 0
offset = 0

def single_trace(inputFiles, label_x, conversions_x, label_y, conversions_y, labels, column, plot_name):

    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)
    ax1.set_xlabel( str(label_x), fontsize=axes_font)
    ax1.set_ylabel( str(label_y), fontsize=axes_font)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)

    for inputFile in inputFiles:
        x = []
        y = []

        data = np.loadtxt(inputFile)
        x = conversions_x[inputFiles.index(inputFile)]*np.array(data[1:,0])
        y = conversions_y[inputFiles.index(inputFile)]*np.array(data[1:,column])
        ax1.plot(x, y,  linewidth=1, label = str(labels[inputFiles.index(inputFile)]))
        ax1.legend(loc=0, fontsize = legend_font)


    fig.subplots_adjust(hspace=sub_hspace)
    fig.subplots_adjust(wspace=sub_wspace)
    fig.tight_layout()
    fig.savefig(str(plot_name)+'.eps',format='eps',transparent = False)
    fig.savefig(str(plot_name)+'.png',format='png',transparent = False)

    return


def gate_dependance(inputFiles, gates, conversions, labels, plot_name):

    fig = plt.figure(figsize=(8,3))
    ax1 = plt.subplot2grid((1,2), (0,0), colspan=1)
    ax2 = plt.subplot2grid((1,2), (0,1), colspan=1)


    ax1.set_xlabel( 'V', fontsize=axes_font)
    ax1.set_ylabel('I ($\mu $ A)', fontsize=axes_font)
    #ax1.tick_params(axis='both', which='major', labelsize=tick_font)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.set_ylim(0, 0.130)





    ax2.set_xlabel( 'V', fontsize=axes_font)
    ax2.set_ylabel('G ($ 2e^2/h$)', fontsize=axes_font)
    #ax2.tick_params(axis='both', which='major', labelsize=tick_font)
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)



    for inputFile in inputFiles:
        Vg = []
        I = []
        G = []
        data = np.loadtxt(inputFile)
        Vg = conversions[inputFiles.index(inputFile)]*np.array(data[1:,0])
        I = np.array(data[1:,4])
        G = 12906.0*np.array(data[1:,3])

        #ax1.plot(Vg, I,  linewidth=2, label = gates[inputFiles.index(inputFile)]+' Vbias:'+ Vs[inputFiles.index(inputFile)])
        #ax2.plot(Vg, G,  linewidth=2, label = gates[inputFiles.index(inputFile)]+' Vbias:'+ Vs[inputFiles.index(inputFile)])
        ax1.plot(Vg, I,  linewidth=2, label = str(labels[inputFiles.index(inputFile)]))
        ax2.plot(Vg, G,  linewidth=2, label = str(labels[inputFiles.index(inputFile)]))
        ax1.legend(loc=0, fontsize = legend_font)
        ax2.legend(loc=0, fontsize = legend_font)

    fig.subplots_adjust(hspace=sub_hspace)
    fig.subplots_adjust(wspace=sub_wspace)
    fig.tight_layout()
    fig.savefig('gates'+str(plot_name)+'.eps',format='eps',transparent = False)
    fig.savefig('gates'+str(plot_name)+'.png',format='png',transparent = False)

    return

def cutter(a, b, x_l, x_r):
    #this function cuts arrays a and b according to the limits x_left and x_right defined for a and return the 2 cutted arrays
    a_new = []
    b_new = []
    for item in a:
        if item<x_r and item>x_l:
            ind = list(a).index(item)
            a_new.append(item)
            b_new.append(b[ind])
    a_new = np.array(a_new)
    b_new = np.array(b_new)
    return a_new, b_new

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def gaus(x,a,x0,sigma, offset):
    return offset + a*np.exp(-(x-x0)**2/(2*sigma**2))

def gaus2peaks(x,a_1, a_2, x0_1, x0_2 ,sigma_1, sigma_2, offset):
    return offset + a_1*np.exp(-(x-x0_1)**2/(2*sigma_1**2)) + a_2*np.exp(-(x-x0_2)**2/(2*sigma_2**2))

def decay(x, A, alpha):
    return A*np.exp(-alpha*x)

def power(x, A, B):
    return A*x**B

def special_function(x, x1, A1,  B1, offset):
    y = offset + A1*np.exp(-B1*(np.sqrt(x)-np.sqrt(x1))**2)*x**(-0.5)
    return y

def special_function_twopeaks(x, x1, A1,  B1, x2, A2,  B2, offset):
    y =  offset + A1*np.exp(-B1*(np.sqrt(x)-np.sqrt(x1))**2)*x**(-0.5) + A2*np.exp(-B2*(np.sqrt(x)-np.sqrt(x2))**2)*x**(-0.5)
    return y

def special_function_twopeaks_locked(x, x1, A1, B1, A2, B2, offset):
    y =  offset + A1*np.exp(-B1*(np.sqrt(x)-np.sqrt(x1))**2)*x**(-0.5) + A2*np.exp(-B2*(np.sqrt(x)-np.sqrt(2*x1))**2)*x**(-0.5)
    return y

def interpolation(x, y):
    xinterp = np.linspace(x[0], x[-1], len(x))
    yinterp = np.interp(xinterp, x, y)
    xinterp = np.array (xinterp)
    yinterp = np.array (yinterp)
    return xinterp, yinterp

def interpol(x, y, n):
    xinterp = np.linspace(x[0], x[-1], n)
    yinterp = np.interp(xinterp, x, y)
    xinterp = np.array (xinterp)
    yinterp = np.array (yinterp)
    return xinterp, yinterp

def residuals(y, window, order):
    #window must be a positive odd integer number
    #order must be less than the the window size
    y_smooth = sci.signal.savgol_filter(y, window, order)
    diff = np.array(y-y_smooth)
    return diff, y_smooth

def fitting_gaus(freq_fft, amplitude_fft, limit_fit, A1, mean1, sigma1):
    freq_fft_fit = []
    amplitude_fft_fit = []



    for item in freq_fft:
                if item<limit_fit and item>0:
                    freq_fft_fit.append(item)
                    ind = list(freq_fft).index(item)
                    amplitude_fft_fit.append(amplitude_fft[ind])

    popt, pcov = curve_fit(gaus, freq_fft_fit, amplitude_fft_fit, p0=[A1, mean1, sigma1])
    A1 = popt[0]
    mean1 = popt[1]
    sigma1 = popt[2]

    return  A1,  mean1, sigma1

def fitting_gaus2peaks(freq_fft, amplitude_fft, limit_fit, A1, A2, mean1, mean2, sigma1, sigma2, offset):
    freq_fft_fit = []
    amplitude_fft_fit = []

    freq_fft_fit, amplitude_fft_fit = cutter( freq_fft, amplitude_fft, 0, limit_fit)
    popt, pcov = curve_fit(gaus2peaks, freq_fft_fit, amplitude_fft_fit, p0=[A1, A2, mean1, mean2, sigma1, sigma2, offset])

    A1 = popt[0]
    A2 = popt[1]
    mean1 = popt[2]
    mean2 = popt[3]
    sigma1 = popt[4]
    sigma2 = popt[5]
    offset = popt[6]

    perr = np.sqrt(np.diag(pcov))
    A1_err = perr[0]
    A2_err = perr[1]
    mean1_err = perr[2]
    mean2_err = perr[3]
    sigma1_err = perr[4]
    sigma2_err = perr[5]
    offset_err = perr[6]

    return  A1, A2, mean1, mean2, sigma1, sigma2, offset, A1_err, A2_err, mean1_err, mean2_err, sigma1_err, sigma2_err, offset_err


def fitting_special_function(freq_fft, amplitude_fft, limit_fit, x1, A1, B1, offset):
    freq_fft_fit = []
    amplitude_fft_fit = []

    freq_fft_fit, amplitude_fft_fit = cutter( freq_fft, amplitude_fft, 0, limit_fit)

    popt, pcov = curve_fit(special_function, freq_fft_fit, amplitude_fft_fit, p0=[x1, A1, B1, offset])
    x1 = popt[0]
    A1 = popt[1]
    B1 = popt[2]
    offset = popt[3]

    perr = np.sqrt(np.diag(pcov))
    x1_err = perr[0]
    A1_err = perr[1]
    B1_err = perr[2]
    offset_err = perr[3]

    return  x1, A1, B1, offset, x1_err, A1_err, B1_err, offset_err

def fitting_special_function_twopeaks(freq_fft, amplitude_fft, limit_fit, x1, A1, B1, x2, A2, B2, offset):
    freq_fft_fit = []
    amplitude_fft_fit = []

    freq_fft_fit, amplitude_fft_fit = cutter( freq_fft, amplitude_fft, 0, limit_fit)

    popt, pcov = curve_fit(special_function_twopeaks, freq_fft_fit, amplitude_fft_fit, p0=[x1, A1, B1, x2, A2, B2, offset])
    x1 = popt[0]
    A1 = popt[1]
    B1 = popt[2]
    x2 = popt[3]
    A2 = popt[4]
    B2 = popt[5]
    offset = popt[6]


    perr = np.sqrt(np.diag(pcov))
    x1_err = perr[0]
    A1_err = perr[1]
    B1_err = perr[2]
    x2_err = perr[3]
    A2_err = perr[4]
    B2_err = perr[5]
    offset_err = perr[6]


    return  x1, A1, B1, x2, A2, B2, offset, x1_err, A1_err, B1_err, x2_err, A2_err, B2_err, offset_err


def fitting_special_function_twopeaks_locked(freq_fft, amplitude_fft, limit_fit, x1, A1, B1, A2, B2, offset):
    freq_fft_fit = []
    amplitude_fft_fit = []

    freq_fft_fit, amplitude_fft_fit = cutter(freq_fft, amplitude_fft, 0, limit_fit)

    popt, pcov = curve_fit(special_function_twopeaks_locked, freq_fft_fit, amplitude_fft_fit,
                           p0=[x1, A1, B1, A2, B2, offset])
    x1 = popt[0]
    A1 = popt[1]
    B1 = popt[2]
    A2 = popt[3]
    B2 = popt[4]
    offset = popt[5]

    perr = np.sqrt(np.diag(pcov))
    x1_err = perr[0]
    A1_err = perr[1]
    B1_err = perr[2]
    A2_err = perr[3]
    B2_err = perr[4]
    offset_err = perr[5]

    x2 = 2 * x1
    x2_err = x1_err

    return x1, A1, B1, x2, A2, B2, offset, x1_err, A1_err, B1_err, x2_err, A2_err, B2_err, offset_err

def residuals_fit(params, x, fft, x1, x2):
    params = A1, B1, A2, B2, offset
    diff = (offset + A1*np.exp(-B1*(np.sqrt(x)-np.sqrt(x1))**2)*x**(-0.5) + A2*np.exp(-B2*(np.sqrt(x)-np.sqrt(x2))**2)*x**(-0.5)) - fft
    return diff



def fitting_decay(x, y, sigma_y, A, alpha):
    # A, alpha are the initial guesses
    # x, y and sigmay need to be a list

    popt, pcov = curve_fit(decay, x, y, p0=[A, alpha], sigma = sigma_y)
    A, alpha = popt
    A_err, alpha_err = np.sqrt(np.diag(pcov))

    return A, alpha, A_err, alpha_err

def fitting_power(x, y, sigma_y, A, B):

    popt, pcov = curve_fit(power, x, y, p0 =[A, B], sigma = sigma_y)
    A, B = popt
    A_err, B_err = np.sqrt(np.diag(pcov))

    return A, B, A_err, B_err


#function to fit the AB residuals
def fit_sinusoidal(x, A, B, C, D):
    y = A*np.cos(B*x-C)+D
    return y


def magnetoconductance(inputFiles, column_B, column_G, windows, orders, labels, plot_name, plot = True , fitting = True):

    if plot is True:


        fig = plt.figure(figsize=(11,18))
        ax1 = plt.subplot2grid((4,2), (0,0), colspan=2)
        ax2 = plt.subplot2grid((4,2), (1,0), colspan=2)
        ax3 = plt.subplot2grid((4,2), (2,0), colspan=2, rowspan =1)

        #ax1.grid()
        ax1.set_xlabel( '$ B (T) $', fontsize=axes_font)
        ax1.set_ylabel('G ($ 2e^2/h$)', fontsize=axes_font)
        ax1.tick_params(axis='both', which='major', labelsize=tick_font)
        ax1.get_xaxis().get_major_formatter().set_useOffset(False)
        #ax1.set_xlim(0.1, 0.2)


        #ax2.grid()
        ax2.set_xlabel( ' $ B (T) $', fontsize=axes_font)
        ax2.set_ylabel('Residuals G ($ 2e^2/h$)', fontsize=axes_font)
        ax2.tick_params(axis='both', which='major', labelsize=tick_font)
        ax2.get_xaxis().get_major_formatter().set_useOffset(False)
        #ax2.set_xlim(0.1, 0.2)


        #ax3.grid()
        ax3.set_xlabel( '$ B^{-1} (T^{-1}) $', fontsize=axes_font)
        ax3.set_ylabel('Amplitude', fontsize=axes_font)
        ax3.tick_params(axis='both', which='major', labelsize=tick_font)
        ax3.get_xaxis().get_major_formatter().set_useOffset(False)
        ax3.set_xlim(0, 1200)

    A1s=[]
    A2s=[]
    mean1s=[]
    mean2s=[]
    sigma1s=[]
    sigma2s=[]
    offsets = []

    A1errs = []
    A2errs = []



    freq_all = []
    fft_all = []

    for inputFile in inputFiles:

        n = inputFiles.index(inputFile)
        data = []
        B = []
        G = []
        B_interp = []
        G_interp = []
        diff = []
        freq_fft = []
        residuals_fft = []
        freq_fft_fit_AB = []
        amplitude_fft_fit_AB = []
        freq_fft_fit = []
        amplitude_fft_fit = []




        shift_ax2 = 0.04
        shift_ax3 = 0.5

        data = np.loadtxt(inputFile, skiprows=1)

        B = np.array(data[1:,column_B])
        G = 12906.0*np.array(data[1:,column_G])


        #flipping in  case of backward scan in field
        if B[0]>B[-1]:
            B = np.flipud(B)
            G = np.flipud(G)


        B_interp = interpolation(B, G)[0]
        G_interp = interpolation(B, G)[1]

        B_step = np.abs(B_interp[0] - B_interp[-1])/len(B_interp)
        window = windows[inputFiles.index(inputFile)]
        order = orders[inputFiles.index(inputFile)]

        diff, G_smooth = residuals(G_interp, window, order)

        residuals_fft = np.fft.fft(diff)  #is a complex number
        amplitude_fft = np.absolute(residuals_fft)
        freq_fft = np.fft.fftfreq(len(B_interp), d = B_step)

        if fitting is True:


            A1 = 1
            mean1 = 171
            sigma1 = 30
            A2 = 0.1
            mean2 = mean1*2
            sigma2 = 50
            offset = 0

            A1, A2, mean1, mean2, sigma1, sigma2, offset, A1err, A2err, mean1_err, mean2_err, sigma1_err, sigma2_err, offset_err = fitting_gaus2peaks(freq_fft, amplitude_fft, limit_fit, A1, A2, mean1, mean2, sigma1, sigma2, offset)

            #A1, A2, mean1, mean2, sigma1, sigma2, offset, A1_err, A2_err, mean1_err, mean2_err, sigma1_err, sigma2_err, offset_err

            freq = np.linspace(0, limit_fit, 2000)
            fit = gaus2peaks(freq, A1, A2, mean1, mean2 ,sigma1, sigma2, offset)
            A1s.append(A1)
            A2s.append(A2)
            mean1s.append(mean1)
            mean2s.append(mean2)
            sigma1s.append(sigma1)
            sigma2s.append(sigma2)
            offsets.append(offset)

            A1errs.append(A1err)
            A2errs.append(A2err)

        #this is a way to cut only some part of the FFT

        if fitting is False:
            A1s.append(0)
            A2s.append(0)
            mean1s.append(0)
            mean2s.append(0)
            sigma1s.append(0)
            sigma2s.append(0)
            offsets.append(0)

            A1errs.append(0)
            A2errs.append(0)

        freq_fft_fit, amplitude_fft_fit = cutter(freq_fft, amplitude_fft, 0, 1600)

        freq_all.append(freq_fft_fit)
        fft_all.append(amplitude_fft_fit)

        if plot is True:


            ax1.plot(B_interp, G_interp,  linewidth=2, label = labels[n])
            #ax1.plot(B_interp, G_smooth, linewidth =2, label = 'smooth')
            ax2.plot(B_interp, diff + shift_ax2 * n ,  linewidth=1, label = labels[n])
            ax3.plot(freq_fft_fit, amplitude_fft_fit + shift_ax3 * n , linewidth=2, label = labels[n])
            #ax3.plot(freq, fit + shift_ax3 * n, color = 'black')
            #ax3.axvline(x = mean1, linestyle = '--', color = 'k')
            #ax3.axvline(x = mean2, linestyle = '--', color = 'y')


            ax1.legend(loc=0)
            ax2.legend(loc=0)
            ax3.legend(loc=0)

        if plot is True:
            fig.subplots_adjust(hspace=sub_hspace)
            fig.subplots_adjust(wspace=sub_wspace)
            fig.tight_layout()
            fig.savefig('magnetoconductance'+str(plot_name)+'.eps',format='eps',transparent = False)
            fig.savefig('magnetoconductance'+str(plot_name)+'.png',format='png',transparent = False)


    return freq_all, fft_all, A1s, mean1s, sigma1s, A2s, mean2s, sigma2s, offsets, A1errs, A2errs

#magnetoconductance with range defined
def magnetoconductance_range(inputFiles, column_B, column_G, windows, orders, labels, plot_name, B_min, B_max, plot = True, fitting = True ):

    if plot is True:


        fig = plt.figure(figsize=(8,12))
        ax1 = plt.subplot2grid((3,2), (0,0), colspan=2)
        ax2 = plt.subplot2grid((3,2), (1,0), colspan=2)
        ax3 = plt.subplot2grid((3,2), (2,0), colspan=2, rowspan =1)

        #ax1.grid()
        ax1.set_xlabel( '$ B (T) $', fontsize=axes_font)
        ax1.set_ylabel('G ($ 2e^2/h$)', fontsize=axes_font)
        ax1.tick_params(axis='both', which='major', labelsize=tick_font)
        ax1.get_xaxis().get_major_formatter().set_useOffset(False)
        ax1.set_xlim(B_min, B_max)
        #ax1.set_ylim(0,1.3)


        #ax2.grid()
        ax2.set_xlabel( ' $ B (T) $', fontsize=axes_font)
        ax2.set_ylabel('Residuals G ($ 2e^2/h$)', fontsize=axes_font)
        ax2.tick_params(axis='both', which='major', labelsize=tick_font)
        ax2.get_xaxis().get_major_formatter().set_useOffset(False)
        ax2.set_xlim(B_min, B_max)


        #ax3.grid()
        ax3.set_xlabel( '$ B^{-1} (T^{-1}) $', fontsize=axes_font)
        ax3.set_ylabel('Amplitude', fontsize=axes_font)
        ax3.tick_params(axis='both', which='major', labelsize=tick_font)
        ax3.get_xaxis().get_major_formatter().set_useOffset(False)
        #ax3.set_xlim(0, 800)

    A1s=[]
    A2s=[]
    mean1s=[]
    mean2s=[]
    sigma1s=[]
    sigma2s=[]
    offsets = []

    A1errs = []
    A2errs = []

    freq_all = []
    fft_all = []

    for inputFile in inputFiles:

        n = inputFiles.index(inputFile)
        data = []
        B = []
        G = []
        B_interp = []
        G_interp = []
        diff = []
        freq_fft = []
        residuals_fft = []
        freq_fft_fit_AB = []
        amplitude_fft_fit_AB = []
        freq_fft_fit = []
        amplitude_fft_fit = []




        shift_ax2 = 0.02
        shift_ax3 = 1

        data = np.loadtxt(inputFile, skiprows = 1)

        B = np.array(data[:,column_B])
        # if B is actually By and this measurement is taken in B2, consider the off set of phi
        phi_offset_deg = 90 + 9.4
        phi_offset_rad = phi_offset_deg * np.pi / 180.0
        B = B/np.sin(phi_offset_rad)


        #G = 12906.0*np.array(data[:,column_G])

        G =  np.array(data[:, column_G])



        #flipping in  case of backward scan in field
        if B[0]>B[-1]:
            B = np.flipud(B)
            G = np.flipud(G)



        B_interp = interpolation(B, G)[0]
        G_interp = interpolation(B, G)[1]

        B_interp, G_interp = cutter(B_interp, G_interp, B_min, B_max)



        B_step = np.abs(B_interp[0] - B_interp[-1])/len(B_interp)
        window = windows[inputFiles.index(inputFile)]
        order = orders[inputFiles.index(inputFile)]

        diff, G_smooth = residuals(G_interp, window, order)



        residuals_fft = np.fft.fft(diff)  #is a complex number
        amplitude_fft = np.absolute(residuals_fft)
        amplitude_fft = sci.signal.savgol_filter(amplitude_fft, 7, 3)
        freq_fft = np.fft.fftfreq(len(B_interp), d = B_step)


        if fitting is True:
            #double FIT
            #limit_fit = 400
            A1 = 1
            mean1 = 171
            sigma1 = 30
            A2 = 0.1
            mean2 = mean1*2
            sigma2 = 50
            offset = 0

            A1, A2, mean1, mean2, sigma1, sigma2, offset, A1err, A2err, mean1err, mean2err, sigma1err, sigma2err, offseterr = fitting_gaus2peaks(freq_fft, amplitude_fft, limit_fit, A1, A2, mean1, mean2, sigma1, sigma2, offset)



            freq = np.linspace(0, limit_fit, 2000)
            fit = gaus2peaks(freq, A1, A2, mean1, mean2 , sigma1, sigma2, offset)

            A1s.append(A1)
            A2s.append(A2)
            mean1s.append(mean1)
            mean2s.append(mean2)
            sigma1s.append(sigma1)
            sigma2s.append(sigma2)
            offsets.append(offset)
            A1errs.append(A1err)
            A2errs.append(A2err)

        if fitting is False:
            A1s.append(0)
            A2s.append(0)
            mean1s.append(0)
            mean2s.append(0)
            sigma1s.append(0)
            sigma2s.append(0)
            offsets.append(0)

            A1errs.append(0)
            A2errs.append(0)



        #this is a way to cut only some part of the FFT


        freq_fft_fit, amplitude_fft_fit = cutter(freq_fft, amplitude_fft, 0, 1600)

        freq_all.append(freq_fft_fit)
        fft_all.append(amplitude_fft_fit)

        if plot is True:


            ax1.plot(B_interp, G_interp,  linewidth=2, label = labels[n])
            ax1.plot(B_interp, G_smooth, linewidth =1, label = 'smooth')
            ax2.plot(B_interp, diff + shift_ax2 * n ,  linewidth=1, label = labels[n])
            ax3.plot(freq_fft_fit, amplitude_fft_fit + shift_ax3 * n , linewidth=1, label = labels[n])
            ax3.set_xlim(0, 400)
            #ax3.plot(freq, fit + shift_ax3 * n, color = 'black')
            #ax3.axvline(x = mean1, linestyle = '--', color = 'k')
            #ax3.axvline(x = mean2, linestyle = '--', color = 'y')


            ax1.legend(loc=0)
            ax2.legend(loc=0)
            ax3.legend(loc=0)

        if plot is True:
            fig.subplots_adjust(hspace=sub_hspace)
            fig.subplots_adjust(wspace=sub_wspace)
            fig.tight_layout()
            fig.savefig( str(plot_name)+ '.eps',format='eps',transparent = False)
            fig.savefig( str(plot_name)+ '.png',format='png',transparent = True)


    return freq_all, fft_all, A1s, mean1s, sigma1s, A2s, mean2s, sigma2s, offsets, A1errs, A2errs

#function that uses the new fitting function for the FFT (the special function)
#magnetoconductance with range defined
def magnetoconductance_range_special(inputFiles, column_B, column_G, windows, orders, labels, plot_name, B_min, B_max, plot = True , fitting = True):

    if plot is True:


        fig = plt.figure(figsize=(11,18))
        ax1 = plt.subplot2grid((4,2), (0,0), colspan=2)
        ax2 = plt.subplot2grid((4,2), (1,0), colspan=2)
        ax3 = plt.subplot2grid((4,2), (2,0), colspan=2, rowspan =1)

        #ax1.grid()
        ax1.set_xlabel( '$ B (T) $', fontsize=axes_font)
        ax1.set_ylabel('G ($ 2e^2/h$)', fontsize=axes_font)
        ax1.tick_params(axis='both', which='major', labelsize=tick_font)
        ax1.get_xaxis().get_major_formatter().set_useOffset(False)
        ax1.set_xlim(B_min, B_max)
        #ax1.set_ylim(0,1.3)


        #ax2.grid()
        ax2.set_xlabel( ' $ B (T) $', fontsize=axes_font)
        ax2.set_ylabel('Residuals G ($ 2e^2/h$)', fontsize=axes_font)
        ax2.tick_params(axis='both', which='major', labelsize=tick_font)
        ax2.get_xaxis().get_major_formatter().set_useOffset(False)
        ax2.set_xlim(B_min, B_max)


        #ax3.grid()
        ax3.set_xlabel( '$ B^{-1} (T^{-1}) $', fontsize=axes_font)
        ax3.set_ylabel('Amplitude', fontsize=axes_font)
        ax3.tick_params(axis='both', which='major', labelsize=tick_font)
        ax3.get_xaxis().get_major_formatter().set_useOffset(False)
        ax3.set_xlim(0, 800)

    x1s=[]
    A1s=[]
    B1s=[]
    x2s = []
    A2s =[]
    B2s = []
    offsets = []

    x1errs = []
    A1errs = []
    B1errs = []
    x2errs = []
    A2errs = []
    B2errs = []
    offseterrs = []

    freq_all = []
    fft_all = []

    for inputFile in inputFiles:

        n = inputFiles.index(inputFile)
        data = []
        B = []
        G = []
        B_interp = []
        G_interp = []
        diff = []
        freq_fft = []
        residuals_fft = []
        freq_fft_fit_AB = []
        amplitude_fft_fit_AB = []
        freq_fft_fit = []
        amplitude_fft_fit = []




        shift_ax2 = 0.02
        shift_ax3 = 3

        data = np.loadtxt(inputFile, skiprows = 1)
        B  = np.array(data[:, column_B])
        #G = np.array(data[:, column_G])
        # if B is actually By and this measurement is taken in B2, consider the off set of phi 90+ 9.4 deg
        #phi_offset_deg = 90 + 9.4
        #phi_offset_rad = phi_offset_deg * np.pi / 180.0
        #B = B / np.sin(phi_offset_rad)
        #G = 12906*np.array(data[:, column_G])
		#column was here 3 for loops, coefficient in front was 12906
		#this is was 5
        G = 12906.0*np.array(data[:,column_G])/10.



        #flipping in  case of backward scan in field
        if B[0]>B[-1]:
            B = np.flipud(B)
            G = np.flipud(G)



        B_interp = interpolation(B, G)[0]
        G_interp = interpolation(B, G)[1]

        B_interp, G_interp = cutter(B_interp, G_interp, B_min, B_max)

        B_step = np.abs(B_interp[0] - B_interp[-1])/len(B_interp)
        window = windows[inputFiles.index(inputFile)]
        order = orders[inputFiles.index(inputFile)]

        diff, G_smooth = residuals(G_interp, window, order)



        residuals_fft = np.fft.fft(diff)  #is a complex number
        amplitude_fft = np.absolute(residuals_fft)
        # adding smoothing
        amplitude_fft = sci.signal.savgol_filter(amplitude_fft, 11, 3)
        freq_fft = np.fft.fftfreq(len(B_interp), d = B_step)


        if fitting is True:

            #double FIT
            #limit_fit = 400
            x1 = 200
            A1 = 1
            B1 = 10
            x2 = 300
            A2 = 0.2
            B2 = 15
            offset = 0


            x1, A1, B1, x2, A2, B2, offset, x1err, A1err, B1err, x2err, A2err, B2err, offseterr = fitting_special_function_twopeaks(freq_fft, amplitude_fft, limit_fit, x1, A1, B1, x2, A2, B2, offset)



            freq = np.linspace(2, limit_fit, 2000)
            fit = special_function_twopeaks(freq, x1, A1,  B1,  x2, A2,  B2, offset)
            x1s.append(x1)
            A1s.append(A1)
            B1s.append(B1)
            x2s.append(x2)
            A2s.append(A2)
            B2s.append(B2)
            offsets.append(offset)


            x1errs.append(x1err)
            A1errs.append(A1err)
            B1errs.append(B1err)
            x2errs.append(x2err)
            A2errs.append(A2err)
            B2errs.append(B2err)
            offseterrs.append(offseterr)


            #this is a way to cut only some part of the FFT

        if fitting is False:
            freq = np.linspace(2, limit_fit, 2000)
            x1s.append(0)
            A1s.append(0)
            B1s.append(0)
            x2s.append(0)
            A2s.append(0)
            B2s.append(0)
            offsets.append(0)


            x1errs.append(0)
            A1errs.append(0)
            B1errs.append(0)
            x2errs.append(0)
            A2errs.append(0)
            B2errs.append(0)
            offseterrs.append(0)




        freq_fft_fit, amplitude_fft_fit = cutter(freq_fft, amplitude_fft, 0, 1600)

        freq_all.append(freq_fft_fit)
        fft_all.append(amplitude_fft_fit)

        if plot is True:


            ax1.plot(B_interp, G_interp,  linewidth=1, label = labels[n])
            ax1.plot(B_interp, G_smooth, linewidth =1, label = 'smooth')
            ax2.plot(B_interp, diff + shift_ax2 * n ,  linewidth=1, label = labels[n])
            ax3.plot(freq_fft_fit, amplitude_fft_fit + shift_ax3 * n , linewidth=1, label = labels[n])
            if fitting is True:
                ax3.plot(freq, fit + shift_ax3 * n, color = 'black')
                ax3.axvline(x = x1, linestyle = '--', color = 'k')
                ax3.axvline(x = x2, linestyle = '--', color = 'k')



            ax1.legend(loc=0)
            ax2.legend(loc=0)
            ax3.legend(loc=0)

        if plot is True:
            fig.subplots_adjust(hspace=sub_hspace)
            fig.subplots_adjust(wspace=sub_wspace)
            fig.tight_layout()
            fig.savefig('magnetoconductance'+str(plot_name)+'.eps',format='eps',transparent = False)
            fig.savefig('magnetoconductance'+str(plot_name)+'.png',format='png',transparent = False)


    return freq_all, fft_all, x1s, A1s, B1s, x2s, A2s, B2s, x1errs, A1errs, B1errs, x2errs, A2errs, B2errs


#define a function that just computes FFT of the signal on a defined range
def magnetoconductance_range_FFT(inputFiles, column_B, column_G, labels, plot_name, B_min, B_max, plot = True ):

    if plot is True:

        fig = plt.figure(figsize=(7,7))
        ax1 = plt.subplot2grid((2,1), (0,0), colspan=1)
        ax2 = plt.subplot2grid((2,1), (1,0), colspan=1)


        #ax1.grid()
        ax1.set_xlabel( '$ B (T) $', fontsize=axes_font)
        ax1.set_ylabel('G ($ 2e^2/h$)', fontsize=axes_font)
        ax1.tick_params(axis='both', which='major', labelsize=tick_font)
        ax1.get_xaxis().get_major_formatter().set_useOffset(False)
        ax1.set_xlim(B_min, B_max)
        #ax1.set_ylim(0,1.3)


        ax2.set_xlabel( '$ B^{-1} (T^{-1}) $', fontsize=axes_font)
        ax2.set_ylabel('Amplitude', fontsize=axes_font)
        ax2.tick_params(axis='both', which='major', labelsize=tick_font)
        ax2.get_xaxis().get_major_formatter().set_useOffset(False)
        ax2.set_xlim(0, 500)

    freq_all = []
    fft_all = []

    for inputFile in inputFiles:

        n = inputFiles.index(inputFile)
        data = []
        B = []
        G = []
        B_interp = []
        G_interp = []
        diff = []
        freq_fft = []
        residuals_fft = []
        freq_fft_fit_AB = []
        amplitude_fft_fit_AB = []
        freq_fft_fit = []
        amplitude_fft_fit = []




        shift_ax2 = 0.02
        shift_ax3 = 1

        data = np.loadtxt(inputFile)
        B = np.array(data[:,column_B])
        G = 12906.0*np.array(data[:,column_G])


        #flipping in  case of backward scan in field
        if B[0]>B[-1]:
            B = np.flipud(B)
            G = np.flipud(G)



        B_interp = interpolation(B, G)[0]
        G_interp = interpolation(B, G)[1]

        B_interp, G_interp = cutter(B_interp, G_interp, B_min, B_max)
        B_step = np.abs(B_interp[0] - B_interp[-1])/len(B_interp)


        conductance_fft = np.fft.fft(G_interp)  #conductance FFT is a complex array
        amplitude_fft = np.absolute(conductance_fft)
        freq_fft = np.fft.fftfreq(len(B_interp), d = B_step)




        #this is a way to cut only some part of the FFT
        freq_fft_fit, amplitude_fft_fit = cutter(freq_fft, amplitude_fft, 0, 1600)


        if plot is True:


            ax1.plot(B_interp, G_interp,  linewidth=2, label = labels[n])
            ax2.plot(freq_fft_fit, amplitude_fft_fit + shift_ax3 * n , linewidth=2, label = labels[n])


            ax1.legend(loc=0)
            ax2.legend(loc=0)
            ax3.legend(loc=0)

        if plot is True:
            fig.subplots_adjust(hspace=sub_hspace)
            fig.subplots_adjust(wspace=sub_wspace)
            fig.tight_layout()
            fig.savefig('magnetoconductance'+str(plot_name)+'.eps',format='eps',transparent = False)
            fig.savefig('magnetoconductance'+str(plot_name)+'.png',format='png',transparent = False)


    return freq_all, fft_all

#trying to nice a decent function for averaging over different number of scan
def magnetconductance_average(inputFiles, column_B, column_G, windows, orders, labels, T, plot_name, A_min, A_center, A_max):
    # A min and A max are given in um2, they are used for calculate the expected limit of AB
    B_1 = 2*2.067e-15*1e12/A_min
    B_center = 2*2.067e-15*1e12/A_center
    B_2 = 2*2.067e-15*1e12/A_max

    f_1 = 1.0/B_1
    f_center = 1./B_center
    f_2 = 1.0/B_2



    parameters = magnetoconductance(inputFiles, column_B, column_G, windows, orders, labels, plot_name, plot = False, fitting =  False)

    freq_all = parameters[0]
    fft_all = parameters[1]

    #need to interpolate each fft cause they have different dimensions sometimes
    n = len(freq_all[0])

    freq_all_interp = []
    fft_all_interp = []

    for index in range(0, len(freq_all)):
        freq_all_interp.append(interpol(freq_all[index], fft_all[index], n)[0])
        fft_all_interp.append(interpol(freq_all[index], fft_all[index], n)[1])

    freq_all_interp = np.array(freq_all_interp)
    fft_all_interp = np.array(fft_all_interp)

    somma = 0
    for index in range(0, len(freq_all)):
        somma = somma + fft_all_interp[index]

    freq_all_average = freq_all_interp[0]

    fft_all_average = somma/(len(freq_all))

    #fitting the average

    #limit_fit = 400
    A1 = 1
    mean1 = 171
    sigma1 = 30
    A2 = 0.1
    mean2 = mean1*2
    sigma2 = 50
    offset = 0
    A1, A2, mean1, mean2, sigma1, sigma2, offset, A1err, A2err, mean1err, mean2err, sigma1err, sigma2err, offseterr = fitting_gaus2peaks(freq_all_average, fft_all_average, limit_fit, A1, A2, mean1, mean2, sigma1, sigma2, offset)



    freq = np.linspace(0, limit_fit, 2000)
    fit = gaus2peaks(freq, A1, A2, mean1, mean2 ,sigma1, sigma2, offset)

    #ax.fill_between(x, 0, 1, where=y > theta, facecolor='green', alpha=0.5, transform=trans)
    #ax.fill_between(x, 0, 1, where=y < -theta, facecolor='red', alpha=0.5, transform=trans)

    #to see the separate contribution
    fit_AB = gaus(freq, A1, mean1, sigma1, offset)
    fit_AAS = gaus(freq, A2, mean2, sigma2, offset)




    fig = plt.figure(figsize=(12,6))
    ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)

    ax1.set_xlabel( '$ B^{-1} (T^{-1})  $', fontsize=axes_font)
    ax1.set_ylabel(' Amplitude ', fontsize=axes_font)
    ax1.tick_params(axis='both', which='major', labelsize=tick_font)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.set_xlim(0,1000)
    ax1.plot(freq_all_average, fft_all_average , linewidth=1)
    ax1.plot(freq, fit , linewidth=2, color = 'black')
    ax1.plot(freq, fit_AB, color = 'grey')
    ax1.plot(freq, fit_AAS, color = 'grey')
    #ax1.axvline( f_1, linestyle = '--', color = 'k')
    #ax1.axvline( f_center, linestyle = '--', color = 'k')
    #ax1.axvline( 2*f_center, linestyle = '--', color = 'k')
    ax1.axvspan(f_1, f_2, facecolor='g', alpha=0.5)
    #ax1.axvspan(f_1*2, f_2*2, facecolor='yellow', alpha=0.5)
    fig.savefig(str(plot_name)+'.eps',format='eps',transparent = False)
    fig.savefig(str(plot_name)+'.png',format='png',transparent = False)


    print('f AB from design:', f_center)
    print('f_AAS from design:', 2*f_center)
    print('limit from design for AB effect', f_1, f_2)
    print('FIT PARAMETERS FOR AB:')
    print('f:', mean1, mean1err)
    print('Amplitude:', A1, A1err)
    print('Sigma:', sigma1, sigma1err)
    print('FIT PARAMETERS FOR AAS:')
    print('f:', mean2, mean2err)
    print('Amplitude:', A2, A2err)
    print('Sigma:', sigma2, sigma2err)
    print('offset:', offset, offseterr)

    return freq_all_average, fft_all_average, A1, A2, mean1, mean2, sigma1, sigma2, offset,  A1err, A2err


#defining function that averages over different FFTs with the new fitting function
def magnetconductance_average_special(inputFiles, column_B, column_G, windows, orders, labels, T, plot_name, A_min, A_center, A_max, B_min, B_max):
    # A min and A max are given in um2, they are used for calculate the expected limit of AB
    #B_min and B_max indicate the range of the data points in which compute the FFT of the data
    B_1 = 2*2.067e-15*1e12/A_min
    B_center = 2*2.067e-15*1e12/A_center
    B_2 = 2*2.067e-15*1e12/A_max

    f_1 = 1.0/B_1
    f_center = 1./B_center
    f_2 = 1.0/B_2

    parameters = magnetoconductance_range_special(inputFiles, column_B, column_G, windows, orders, labels, plot_name, B_min, B_max, plot = False , fitting = False)
    freq_all = parameters[0]
    fft_all = parameters[1]

    #need to interpolate each fft cause they have different dimensions sometimes
    n = len(freq_all[0])

    freq_all_interp = []
    fft_all_interp = []

    for index in range(0, len(freq_all)):
        freq_all_interp.append(interpol(freq_all[index], fft_all[index], n)[0])
        fft_all_interp.append(interpol(freq_all[index], fft_all[index], n)[1])

    freq_all_interp = np.array(freq_all_interp)
    fft_all_interp = np.array(fft_all_interp)

    somma = 0
    for index in range(0, len(freq_all)):
        somma = somma + fft_all_interp[index]

    freq_all_average = freq_all_interp[0]
    fft_all_average = somma/(len(freq_all))

    #fitting the average with the special function
    x1 = f_center
    A1 = 10
    B1 = 0.3
    x2 = 2*f_center
    A2 = 5
    B2 = 0.4
    offset = 0


    x1, A1, B1, x2, A2, B2, offset, x1err, A1err, B1err, x2err, A2err, B2err, offset_err = fitting_special_function_twopeaks(freq_all_average, fft_all_average, limit_fit, x1, A1, B1,  x2, A2, B2, offset)


    freq = np.linspace(2, limit_fit, 2000)
    fit = special_function_twopeaks(freq, x1, A1,  B1, x2, A2,  B2, offset)

    #to see the separate contribution
    fit_AB = special_function(freq, x1, A1, B1, offset)
    fit_AAS = special_function(freq, x2, A2, B2, offset)

    #plotting
    fig = plt.figure(figsize=(9,6))
    ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)

    ax1.set_xlabel( '$ B^{-1} (T^{-1})  $', fontsize=axes_font)
    ax1.set_ylabel(' Amplitude ', fontsize=axes_font)
    ax1.tick_params(axis='both', which='major', labelsize=tick_font)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.set_xlim(0,400)
    ax1.plot(freq_all_average, fft_all_average , linewidth=1)
    ax1.plot(freq, fit , linewidth=2, color = 'black')
    ax1.plot(freq, fit_AB, linewidth =1, color ='grey')
    ax1.plot(freq, fit_AAS, linewidth =1, color ='grey')
    ax1.axvline( x1, linestyle = '--', color = 'k', label='AB from fit')
    ax1.axvline( x2, linestyle = '--', color = 'k', label='AAS from fit')
    ax1.axvspan(f_1, f_2, facecolor='g', alpha=0.5)
    ax1.legend(loc=0)


    fig.savefig( str(plot_name) +'.eps',format='eps',transparent = False)
    fig.savefig(str(plot_name) + '.png',format='png',transparent = False)


    print('f AB from design:', round(f_center, 1), 'corrisponding to a periodicity of:', round(1/f_center, 4), 'T')
    print('f_AAS from design:', round(2*f_center, 1), 'corrisponding to a periodicity of:', round(1/(2*f_center),4) , 'T')
    print('limit from design for AB effect', round(f_1,1) , round(f_2,1))
    print('FIT PARAMETERS FOR AB:')
    print('f:', round(x1,1),'pm', round(x1err,1))
    print('Amplitude:', round(A1,2),'pm', round(A1err,2))
    print('B parameter:', round(B1,4),'pm', round(B1err,4))
    print('FIT PARAMETERS FOR AAS:')
    print('f:', round(x2, 1),'pm', round(x2err, 1))
    print('Amplitude:', round(A2,2),'pm', round(A2err, 2))
    print('B parameter:', round(B2, 4),'pm', round(B2err,4))
    print('offset:', round(offset,2),'pm', round(offset_err,2))

    return freq_all_average, fft_all_average, x1, A1, B1, x2, A2, B2, offset, x1err, A1err, B1err, x2err, A2err, B2err, offset_err

def magnetconductance_average_special_singlepeak(inputFiles, column_B, column_G, windows, orders, labels, T, plot_name, A_min, A_center, A_max, B_min, B_max):
    # A min and A max are given in um2, they are used for calculate the expected limit of AB
    #B_min and B_max indicate the range of the data points in which compute the FFT of the data
    B_1 = 2*2.067e-15*1e12/A_min
    B_center = 2*2.067e-15*1e12/A_center
    B_2 = 2*2.067e-15*1e12/A_max


    f_1 = 1.0/B_1
    f_center = 1./B_center
    f_2 = 1.0/B_2
    #print f_1, f_2


    parameters = magnetoconductance_range_special(inputFiles, column_B, column_G, windows, orders, labels, plot_name, B_min, B_max, plot = False , fitting = False)

    freq_all = parameters[0]
    fft_all = parameters[1]

    #need to interpolate each fft cause they have different dimensions sometimes
    n = len(freq_all[0])

    freq_all_interp = []
    fft_all_interp = []

    for index in range(0, len(freq_all)):
        freq_all_interp.append(interpol(freq_all[index], fft_all[index], n)[0])
        fft_all_interp.append(interpol(freq_all[index], fft_all[index], n)[1])

    freq_all_interp = np.array(freq_all_interp)
    fft_all_interp = np.array(fft_all_interp)

    somma = 0
    for index in range(0, len(freq_all)):
        somma = somma + fft_all_interp[index]

    freq_all_average = freq_all_interp[0]
    #fft_all_average = somma/(len(freq_all) + 1)
    fft_all_average = somma/(len(freq_all) )

    #fitting the average with the special function

    #x1 = 60
    x1 = f_center
    A1 = 1
    B1 = 0.3
    offset = 0


    x1, A1, B1, offset, x1err, A1err, B1err, offset_err = fitting_special_function(freq_all_average, fft_all_average, limit_fit, x1, A1, B1, offset)


    freq = np.linspace(2, limit_fit, 2000)
    fit = special_function(freq, x1, A1,  B1, offset)



    fig = plt.figure(figsize=(12,6))
    ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)

    ax1.set_xlabel( '$ B^{-1} (T^{-1})  $', fontsize=axes_font)
    ax1.set_ylabel(' Amplitude ', fontsize=axes_font)
    ax1.tick_params(axis='both', which='major', labelsize=tick_font)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.set_xlim(0, 800)
    ax1.plot(freq_all_average, fft_all_average , linewidth=1)
    ax1.plot(freq, fit , linewidth=2, color = 'black')
    ax1.axvline( x1, linestyle = '--', color = 'k', label='AB from fit')
    ax1.axvspan(f_1, f_2, facecolor='g', alpha=0.5)
    ax1.legend(loc=0)

    print('f AB from design:', f_center)
    print('f_AAS from design:', 2*f_center)
    print('limit from design for AB effect', f_1, f_2)
    print('FIT PARAMETERS FOR AB:')
    print('f:', x1, x1err)
    print('Amplitude:', A1, A1err)
    print('B parameter:', B1, B1err)
    print('offset:', offset)


    fig.savefig(str(plot_name) + '.eps',format='eps', transparent = False)
    fig.savefig(str(plot_name) + '.png',format='png', transparent = False)

    return freq_all_average, fft_all_average, x1, A1, B1, offset, x1err, A1err, B1err, offset_err



#defining fitting function with defined peak positions for the first and second harmonic
def magnetconductance_average_special_fixedpeaks(inputFiles, windows, orders, labels, T, plot_name, A_min, A_center, A_max, B_min, B_max):
    # A min and A max are given in um2, they are used for calculate the expected limit of AB
    #B_min and B_max indicate the range of the data points in which compute the FFT of the data
    B_1 = 2*2.067e-15*1e12/A_min
    B_center = 2*2.067e-15*1e12/A_center
    B_2 = 2*2.067e-15*1e12/A_max
    f_1 = 1.0/B_1
    f_center = 1./B_center
    f_2 = 1.0/B_2


    parameters = magnetoconductance_range_special(inputFiles, windows, orders, labels, plot_name, B_min, B_max, plot = False , fitting = False)
    freq_all = parameters[0]
    fft_all = parameters[1]

    #need to interpolate each fft cause they have different dimensions sometimes
    n = len(freq_all[0])

    freq_all_interp = []
    fft_all_interp = []

    for index in range(0, len(freq_all)):
        freq_all_interp.append(interpol(freq_all[index], fft_all[index], n)[0])
        fft_all_interp.append(interpol(freq_all[index], fft_all[index], n)[1])

    freq_all_interp = np.array(freq_all_interp)
    fft_all_interp = np.array(fft_all_interp)

    somma = 0
    for index in range(0, len(freq_all)):
        somma = somma + fft_all_interp[index]

    freq_all_average = freq_all_interp[0]
    fft_all_average = somma/(len(freq_all))

    #fitting the average with the special function  fixing the 2 peaks positions
    x1 = f_center
    x2 = 2*f_center

    A1 = 10
    B1 = 0.3
    A2 = 5.0
    B2 = 0.4
    offset = 0

    p_guess =  A1, B1, A2, B2, offset
    optparameter = op.leastsq(residuals_fit, p_guess, args = (freq_all_average, fft_all_average, x1, x2), full_output = 1)[0]
    cov =  op.leastsq(residuals_fit, p_guess, args = (freq_all_average, fft_all_average, x1, x2), full_output = 1)[1]
    #aggiungere varianza e continuare a sistemare

    print('optimal parameter', optparameter)
    print('covariance', cov)

    freq = np.linspace(2, limit_fit, 2000)

    fit = special_function_twopeaks(freq, x1, A1,  B1, x2, A2,  B2, offset)

    #to see the separate contribution
    fit_AB = special_function(freq, x1, A1, B1, offset)
    fit_AAS = special_function(freq, x2, A2, B2, offset)

    #plotting
    fig = plt.figure(figsize=(12,6))
    ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)

    ax1.set_xlabel( '$ B^{-1} (T^{-1})  $', fontsize=axes_font)
    ax1.set_ylabel(' Amplitude ', fontsize=axes_font)
    ax1.tick_params(axis='both', which='major', labelsize=tick_font)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.set_xlim(0,1200)
    ax1.plot(freq_all_average, fft_all_average , linewidth=1)
    ax1.plot(freq, fit , linewidth=2, color = 'black')
    ax1.plot(freq, fit_AB, linewidth =1, color ='grey')
    ax1.plot(freq, fit_AAS, linewidth =1, color ='grey')
    ax1.axvline( x1, linestyle = '--', color = 'k', label='AB from fit')
    ax1.axvline( x2, linestyle = '--', color = 'k', label='AAS from fit')
    ax1.axvspan(f_1, f_2, facecolor='g', alpha=0.5)
    ax1.legend(loc=0)


    fig.savefig( str(plot_name) +'.eps',format='eps',transparent = False)
    fig.savefig(str(plot_name) + '.png',format='png',transparent = False)

    print('f AB from design:', f_center)
    print('f_AAS from design:', 2*f_center)
    print('limit from design for AB effect', f_1, f_2)
    print('FIT PARAMETERS FOR AB:')
    print('f:', x1)
    print('Amplitude:', A1)
    print('B parameter:', B1)
    print('FIT PARAMETERS FOR AAS:')
    print('f:', x2)
    print('Amplitude:', A2)
    print('B parameter:', B2)
    print('offset:', offset)

    return freq_all_average, fft_all_average, x1, A1, B1, x2, A2, B2, offset


def colorplot_data(inputFile, label_x, conversion_x, label_y, conversion_y,  conversion_z, column, plot_name, points_list):
    #column =  3 conductance
    #column = 4 current
    #point list is a list of single points that need to be plotted
    #point list i.e. [(0,0), (1,0)] in the raw data - with no conversions

    #loading data file
    data = np.loadtxt(inputFile)

    #extract values from metadata
    metadat=open(inputFile[:-4]+'.meta.txt','r')
    metadat=metadat.readlines()

    xpoints=int(metadat[0])
    xstart=float(metadat[1])
    xend=float(metadat[2])
    x = conversion_x*np.linspace(xstart,xend,xpoints)

    ypoints=int(metadat[4])
    yend=float(metadat[5])
    ystart=float(metadat[6])
    y = conversion_y * np.linspace(ystart,yend,ypoints)


    z = conversion_z * np.array(data[:,column])
    Z = np.reshape(z, [  len(y), len(x),])



    fig2D = plt.figure()
    im = plt.imshow(Z, cmap=plt.cm.seismic, interpolation='none', origin='lower', extent=[x[1],x[-1],y[0],y[-1]], aspect="auto")

    cbar = plt.colorbar(im, orientation='vertical')
    plt.xlabel(str(label_x),fontsize=axes_font)
    plt.ylabel(str(label_y), fontsize=axes_font)

    if len(points_list)>0:
        for point in points_list:
            x_p = point[0]
            y_p = point[1]
            plt.plot(conversion_x*x_p, conversion_y * y_p,  'o', color = 'yellow')

    if column==3:
            cbar.set_label('I ($ \mu A $)',size=axes_font)
    if column==4:
            cbar.set_label('G ( $ 2e^2 /h $)', size=axes_font)


    fig2D.savefig(str(plot_name)+'.png', bbox_inches='tight')

    return


def cotunneling(inputFile_bias, inputFile_AB, conversions_bias_x, conversions_bias_y, label_bias, conversions_AB_x, conversions_AB_y, label_AB,
                B_min, B_max, G_min, G_max, plot_name):

    fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot2grid((1,2), (0,0), colspan=1)
    ax2 = plt.subplot2grid((1,2), (0,1), colspan=1)


    ax1.set_xlabel( '$ V_{SD} $ (mV)', fontsize=axes_font)
    ax1.set_ylabel('G ($ 2e^2 /h $ )', fontsize=axes_font)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)




    ax2.set_xlabel( 'B (T)', fontsize=axes_font)
    ax2.set_ylabel('G ($ 2e^2/h$)', fontsize=axes_font)
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    ax2.set_xlim(B_min, B_max)
    ax2.set_ylim(G_min, G_max)



    Vbias = []
    G = []
    data = np.loadtxt(inputFile_bias)
    Vbias = conversions_bias_x*np.array(data[1:,0])
    #G = conversions_bias_y*np.array(data[:,3])
    Lx = np.array(data[1:, 5])
    Rs = 7272
    Imeas = 1e6
    amplitude = 10e-6
    G = 12906.0 * (Lx / Imeas) / ((amplitude / np.sqrt(2.)) - Lx * Rs / Imeas)

    ax1.set_ylim(0, max(G)*1.2)


    B = []
    G_AB = []
    data_AB = np.loadtxt(inputFile_AB)
    B = conversions_AB_x*np.array(data_AB[1:,0])
    #G_AB = conversions_AB_y*np.array(data_AB[:,3])
    Lx = np.array(data_AB[1:, 5])
    Rs = 7272
    Imeas = 1e6
    amplitude = 10e-6
    G_AB = 12906.0 * (Lx / Imeas) / ((amplitude / np.sqrt(2.)) - Lx * Rs / Imeas)



    ax1.plot(Vbias, G,  linewidth=1, label = str(label_bias))
    ax2.plot(B, G_AB,  linewidth=1, label = str(label_AB))
    ax1.legend(loc=0, fontsize = legend_font)
    ax2.legend(loc=0, fontsize = legend_font)

    fig.subplots_adjust(hspace=sub_hspace)
    fig.subplots_adjust(wspace=sub_wspace)
    fig.tight_layout()
    fig.savefig(str(plot_name)+'.eps',format='eps',transparent = False)
    fig.savefig(str(plot_name)+'.png',format='png',transparent = False)

    return B, G_AB, Vbias, G


def WAL_plot(inputFile, conversion_x, conversion_y, conversion_z, label_x, label_y, label_z, y_values, plot_name):

    #y_values is a list of the values that you want to plot - not converted



    fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot2grid((1,2), (0,0), colspan=1)
    ax2 = plt.subplot2grid((1,2), (0,1), colspan=1)


    ax1.set_xlabel( str(label_x) , fontsize=axes_font)
    ax1.set_ylabel(str(label_z), fontsize=axes_font)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)

    ax2.set_xlabel( str(label_x), fontsize=axes_font)
    ax2.set_ylabel(str(label_z), fontsize=axes_font)
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)

    length = []


    for y in y_values:

        ind = list(y_values).index(y)
        data = np.loadtxt(inputFile)
        data=data[data[:,1]==y]

        x = data[1: , 0]
        z = data[1: , 3]



        #correction for series resistance
        z =np.array(z)
        z = z /(1. - R_c*z)

        x = conversion_x*np.array(x)
        z = conversion_z*np.array(z)

        n = len(x)
        length.append(n)


        ax1.plot(x, z,  linewidth=1, label =  str(label_y) + str(np.round(y*conversion_y, 2)) + str(' V'))
        #ax1.legend(loc=0, fontsize = legend_font)


    n_min = min(length)
    z_tot = np.zeros(n_min)

    for y in y_values:
        xinterp = []
        zinterp = []

        ind = list(y_values).index(y)
        data = np.loadtxt(inputFile)
        data=data[data[:,1]==y]

        x = data[1: , 0]
        z = data[1: , 3]

        #correction for series resistance
        z =np.array(z)
        z = z /(1. - R_c*z)

        x = conversion_x*np.array(x)
        z = conversion_z*np.array(z)

        #flipping in  case of backward scan in field
        if x[0]>x[-1]:
            x = np.flipud(x)
            z = np.flipud(z)


        xvals = np.linspace(x[0], x[-1], n_min)
        zinterp = np.interp(xvals, x, z)

        #average the first 50 and last 50 points
        z_mean = (np.mean(zinterp[0:50])+np.mean(zinterp[-51:-1]))/2.0
        zinterp = zinterp-z_mean

        z_tot =  z_tot + zinterp

    z_average =  z_tot/len(y_values)

    #ax2.plot(x, z,  linewidth=1, label =  str(label_y) + str(np.round(y*conversion_y, 2)) + str(' V'))
    ax2.plot(xvals, z_average,  linewidth=1, label =  str(label_y) + str(np.round(y*conversion_y, 2)) + str(' V'))
    #ax2.legend(loc=0, fontsize = legend_font)









    fig.subplots_adjust(hspace=sub_hspace)
    fig.subplots_adjust(wspace=sub_wspace)
    fig.tight_layout()
    fig.savefig(str(plot_name)+'.eps',format='eps',transparent = False)
    fig.savefig(str(plot_name)+'.png',format='png',transparent = False)


    return


def extract(filename, x_col, y_col):
    # x_col and y_col are the # of the column in the data
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)
    x, y = data[:, x_col], data[:, y_col]
    return x, y


def colorplot(inputFile,
              col_x=1, label_x='x', conversion_x=1,
              col_y=0, label_y='y', conversion_y=1,
              col_z=6, label_z='z', conversion_z=1,
              plot_name='colorplot', points_list=None,
              horizontal_linecuts= None,
              vertical_linecuts= None,
              x_lims = False,
              y_lims = False):


    # inputFile is the only mandatory argument i.e. 'folder/../example.dat'
    # others are options
    # col_x: integer, refers to column to use as y
    # col_y: integer, refers to column to use as x
    # labels are strings
    # point list i.e. [(0,0), (1,0)] in the raw data - with no conversions
    # horizontal linecuts is a list of y_cuts


    # loading data file
    data = np.loadtxt(inputFile, skiprows=1)
    x = conversion_x * np.array(data[:, col_x])  # fast scan direction (usually second column)
    y = conversion_y * np.array(data[:, col_y])  # slow scan direction (usually first column)
    z = conversion_z * np.array(data[:, col_z])  # to plot

    idx = np.nonzero(x == x[0])[0]
    len_x = idx[1]
    x = x.reshape((-1, len_x))
    y = y.reshape((-1, len_x))
    z = z.reshape((-1, len_x))
    # data as array x, y
    flatx = x[0, :]
    flaty = y[:, 0]
    # print(flatx)
    # print(flaty)

    # fig2D = plt.figure(figsize = (8,14))
    fig2D = plt.figure(figsize=(8, 8))
    if x_lims is not False:
        plt.xlim(x_lims[0], x_lims[1])
    if y_lims is not False:
        plt.ylim(y_lims[0], y_lims[1])
    im = plt.imshow(z, cmap=plt.cm.seismic, interpolation='none', origin='lower',
                    extent=[flatx[0], flatx[-1], flaty[0], flaty[-1]], aspect="auto")
    cbar = plt.colorbar(im, orientation='vertical')
    plt.xlabel(label_x, fontsize=axes_font)
    plt.ylabel(label_y, fontsize=axes_font)
    cbar.set_label(label_z, size=axes_font)


    if points_list is not None:
        for point in points_list:
            plt.plot(conversion_x * point[0], conversion_y * point[1], 'o', color='yellow')
            print('Point at:', point[0], ',', point[1])
    fig2D.savefig(str(plot_name) + '.png', bbox_inches='tight')

    # horizontal linecuts
    if horizontal_linecuts is not None:

        hcut = plt.figure()
        plt.xlabel(label_x, fontsize=axes_font)
        plt.ylabel(label_z, size=axes_font)
        for y_cut in horizontal_linecuts:
            y_cut_real = find_nearest(flaty, y_cut)
            data_hcut = data[data[:, 0] == y_cut_real]
            z_cut = conversion_z * np.array(data_hcut[:, col_z])
            x_cut = conversion_x * np.array(data_hcut[:, col_x])
            plt.plot(x_cut, z_cut, label=str(y_cut_real))

        plt.legend(loc=0)

    # vertical linecuts
    if vertical_linecuts is not None:
        vcut = plt.figure()
        plt.xlabel(label_y, fontsize=axes_font)
        plt.ylabel(label_z, size=axes_font)
        for x_cut in vertical_linecuts:
            x_cut_real = find_nearest(flatx, x_cut)
            data_vcut = data[data[:, 1] == x_cut_real]
            z_cut = conversion_z * np.array(data_vcut[:, col_z])
            y_cut = conversion_x * np.array(data_vcut[:, col_y])
            plt.plot(y_cut, z_cut, label=str(x_cut_real))

        plt.legend(loc=0)

    return flatx, flaty, z

#adjusting the max and minima of the color plot
def colorplot_special(inputFile, size_x = 6, size_y = 14,
              col_x=1, label_x='x', conversion_x=1.0,
              col_y=0, label_y='y', conversion_y=1.0,
              col_z=6, label_z='z', conversion_z=1.0,
              plot_name='colorplot', points_list=None,
              horizontal_linecuts= None,
              vertical_linecuts= None,
              x_lims = False,
              y_lims = False,
              color_min = 0,
              color_max = 1):

    # inputFile is the only mandatory argument i.e. 'folder/../example.dat'
    # others are options
    # col_x: integer, refers to column to use as y
    # col_y: integer, refers to column to use as x
    # labels are strings
    # point list i.e. [(0,0), (1,0)] in the raw data - with no conversions
    # horizontal linecuts is a list of y_cuts

    # loading data file for the first time
    data = np.loadtxt(inputFile, skiprows=1)
    x = conversion_x * np.array(data[:, col_x])  # fast scan direction (usually second column)
    y = conversion_y * np.array(data[:, col_y])  # slow scan direction (usually first column)
    z = conversion_z * np.array(data[:, col_z])  # to plot
    idx = np.nonzero(x == x[0])[0]
    len_x = idx[1]

    #looping until x, y, and z cam be reshaped correctly.
    while (len(x) % len_x !=0):
        time.sleep(0.3)
        data = np.loadtxt(inputFile, skiprows=1)
        x = conversion_x * np.array(data[:, col_x])  # fast scan direction (usually second column)
        y = conversion_y * np.array(data[:, col_y])  # slow scan direction (usually first column)
        z = conversion_z * np.array(data[:, col_z])  # to plot
        idx = np.nonzero(x == x[0])[0]
        len_x = idx[1]


    x = x.reshape((-1, len_x))
    y = y.reshape((-1, len_x))
    z = z.reshape((-1, len_x))
    flatx = x[0, :]
    flaty = y[:, 0]

    if horizontal_linecuts is not None and vertical_linecuts is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(size_x, size_y), nrows=3, ncols=1, squeeze=True)

        if x_lims is not False:
            ax1.set_xlim(x_lims[0], x_lims[1])
        if y_lims is not False:
            ax1.set_ylim(y_lims[0], y_lims[1])
        im = ax1.imshow(z, cmap=plt.cm.RdBu, norm=LogNorm(vmin=z.min(), vmax=z.max()),  interpolation='none', origin='lower',
                        extent=[flatx[0], flatx[-1], flaty[0], flaty[-1]], aspect="auto", vmin = color_min, vmax = color_max)
        cbar = fig.colorbar(im, ax=ax1, orientation='vertical', aspect = 10, shrink = 0.8 )

        cbar.ax.tick_params(which='both', direction='in', width=1)
        cbar.set_label(label_z, labelpad=-2)
        cbar.set_label(label_z)
        ax1.set_xlabel(label_x)
        ax1.set_ylabel(label_y)
        ax1.set_title(inputFile)
        if points_list is not None:
            for point in points_list:
                ax1.plot(conversion_x * point[0], conversion_y * point[1], '*', color='yellow', markersize  = 10)
                print('First point at:', point[0], ',', point[1])
        # horizontal linecuts
        if horizontal_linecuts is not None:
            ax2.set_xlabel(label_x)
            ax2.set_ylabel(label_z)
            cbar2 = plt.colorbar(im, ax=ax2, orientation='vertical')
            cbar2.remove()
            for y_cut in horizontal_linecuts:
                y_cut_real = find_nearest(flaty, y_cut)
                data_hcut = data[data[:, col_y] == y_cut_real]
                z_cut = conversion_z * np.array(data_hcut[:, col_z])
                x_cut = conversion_x * np.array(data_hcut[:, col_x])
                ax2.plot(x_cut, z_cut, label=str(np.round(y_cut_real,2)))
            if x_lims is not False:
                ax2.set_xlim(x_lims[0], x_lims[1])
            ax2.legend(loc=0)
        else:
            ax2.remove()
        # vertical linecuts
        if vertical_linecuts is not None:
            ax3.set_xlabel(label_y)
            ax3.set_ylabel(label_z)
            cbar3 = plt.colorbar(im, ax=ax3, orientation='vertical')
            cbar3.remove()
            for x_cut in vertical_linecuts:
                x_cut_real = find_nearest(flatx, x_cut)
                data_vcut = data[data[:, col_x] == x_cut_real]
                z_cut = conversion_z * np.array(data_vcut[:, col_z])
                y_cut = conversion_x * np.array(data_vcut[:, col_y])
                ax3.plot(y_cut, z_cut, label=str(np.round(x_cut_real)))
            if y_lims is not False:
                ax3.set_xlim(y_lims[0], y_lims[1])
            ax3.legend(loc=0)
        else:
            ax3.remove()
        #fig.tight_layout()
        fig.savefig(plot_name.strip('.txt') + '.png', format = 'png', bbox_inches='tight')
        fig.savefig(plot_name.strip('.txt') + '.pdf', format = 'pdf', bbox_inches='tight')

    if horizontal_linecuts is not None and vertical_linecuts is None:
        fig, (ax1, ax2) = plt.subplots(figsize=(size_x, size_y), nrows=2, ncols=1, squeeze=True)

        if x_lims is not False:
            ax1.set_xlim(x_lims[0], x_lims[1])
        if y_lims is not False:
            ax1.set_ylim(y_lims[0], y_lims[1])
        im = ax1.imshow(z, cmap=plt.cm.RdBu,
                        interpolation = 'none',
                        #norm=LogNorm(vmin=z.min(), vmax=z.max()),
                        origin='lower',
                        extent=[flatx[0], flatx[-1], flaty[0], flaty[-1]],
                        aspect="auto"
                        #,vmin=color_min, vmax=color_max
                        )
        cbar = fig.colorbar(im, ax=ax1, orientation='vertical', aspect=10, shrink=0.8)

        cbar.ax.tick_params( which='both', direction='in', width=1)
        cbar.set_label(label_z,  labelpad=0)
        cbar.set_label(label_z)
        #ax1.set_xlabel(label_x)
        ax1.set_ylabel(label_y)
        #ax1.set_title(inputFile)
        if points_list is not None:
            for point in points_list:
                ax1.plot(conversion_x * point[0], conversion_y * point[1], '*', color='yellow', markersize=10)
                print('First point at:', point[0], ',', point[1])
        # horizontal linecuts
        if horizontal_linecuts is not None:
            ax2.set_xlabel(label_x)
            ax2.set_ylabel(label_z)
            cbar2 = plt.colorbar(im, ax=ax2, orientation='vertical')
            cbar2.remove()
            for y_cut in horizontal_linecuts:
                y_cut_real = find_nearest(flaty, y_cut)
                data_hcut = data[data[:, col_y] == y_cut_real]
                z_cut = conversion_z * np.array(data_hcut[:, col_z])
                x_cut = conversion_x * np.array(data_hcut[:, col_x])
                ax2.plot(x_cut, z_cut, label=str(np.round(y_cut_real, 2)))
            if x_lims is not False:
                ax2.set_xlim(x_lims[0], x_lims[1])
            ax2.legend(loc=0)
        else:
            ax2.remove()
        #plt.tight_layout()
        fig.savefig(plot_name.strip('.txt') + '.png', format='png', bbox_inches='tight')
        fig.savefig(plot_name.strip('.txt') + '.pdf', format='pdf', bbox_inches='tight')

    if horizontal_linecuts is None and vertical_linecuts is not None:
        fig, (ax1, ax3) = plt.subplots(figsize=(size_x, size_y), nrows=2, ncols=1, squeeze=True)

        if x_lims is not False:
            ax1.set_xlim(x_lims[0], x_lims[1])
        if y_lims is not False:
            ax1.set_ylim(y_lims[0], y_lims[1])
        im = ax1.imshow(z, cmap=plt.cm.RdBu, norm=LogNorm(vmin=z.min(), vmax=z.max()), interpolation='none',
                        origin='lower',
                        extent=[flatx[0], flatx[-1], flaty[0], flaty[-1]], aspect="auto", vmin=color_min,
                        vmax=color_max)
        cbar = fig.colorbar(im, ax=ax1, orientation='vertical', aspect=10, shrink=0.8)

        cbar.ax.tick_params( which='both', direction='in', width=1)
        cbar.set_label(label_z,  labelpad=-2)
        cbar.set_label(label_z)
        ax1.set_xlabel(label_x)
        ax1.set_ylabel(label_y)
        ax1.set_title(inputFile)
        if points_list is not None:
            for point in points_list:
                ax1.plot(conversion_x * point[0], conversion_y * point[1], '*', color='yellow', markersize=10)
                print('First point at:', point[0], ',', point[1])
        # vertical linecuts
        if vertical_linecuts is not None:
            ax3.set_xlabel(label_y)
            ax3.set_ylabel(label_z)
            cbar3 = plt.colorbar(im, ax=ax3, orientation='vertical')
            cbar3.remove()
            for x_cut in vertical_linecuts:
                x_cut_real = find_nearest(flatx, x_cut)
                data_vcut = data[data[:, col_x] == x_cut_real]
                z_cut = conversion_z * np.array(data_vcut[:, col_z])
                y_cut = conversion_x * np.array(data_vcut[:, col_y])
                ax3.plot(y_cut, z_cut, label=str(np.round(x_cut_real)))
            if y_lims is not False:
                ax3.set_xlim(y_lims[0], y_lims[1])
            ax3.legend(loc=0)
        else:
            ax3.remove()
        # fig.tight_layout()
        fig.savefig(plot_name.strip('.txt') + '.png', format='png', bbox_inches='tight')
        fig.savefig(plot_name.strip('.txt') + '.pdf', format='pdf', bbox_inches='tight')

    if horizontal_linecuts is None and vertical_linecuts is  None:
        fig, ax1 = plt.subplots(figsize=(size_x, size_y), nrows=1, ncols=1, squeeze=True)

        if x_lims is not False:
            ax1.set_xlim(x_lims[0], x_lims[1])
        if y_lims is not False:
            ax1.set_ylim(y_lims[0], y_lims[1])
        im = ax1.imshow(z, cmap=plt.cm.RdBu,
                        interpolation='none',
                        norm=LogNorm(vmin=z.min(), vmax=z.max()),
                        origin='lower',
                        extent=[flatx[0], flatx[-1], flaty[0], flaty[-1]], aspect="auto"
                        , vmin=color_min, vmax=color_max
                        )
        cbar = fig.colorbar(im, ax=ax1, orientation='vertical', aspect=10, shrink=0.8)

        cbar.ax.tick_params( which='both', direction='in', width=1)
        cbar.set_label(label_z, labelpad=-2)
        cbar.set_label(label_z)
        ax1.set_xlabel(label_x)
        ax1.set_ylabel(label_y)
        plt.tight_layout()
        #ax1.set_title(inputFile)
        if points_list is not None:
            for point in points_list:
                ax1.plot(conversion_x * point[0], conversion_y * point[1], '*', color='yellow', markersize=10)
                print('First point at:', point[0], ',', point[1])
                # fig.tight_layout()
        fig.savefig(plot_name.strip('.txt') + '.png', format='png', bbox_inches='tight')
        fig.savefig(plot_name.strip('.txt') + '.pdf', format='pdf', bbox_inches='tight')
    return flatx, flaty, z


#function for extracting and plotting a horizontal linecut of a 2D data plot
def horizontal_linecut(inputFile,
                       col_x, label_x, conversion_x,
                       col_y, label_y, conversion_y,
                       col_z, label_z, conversion_z,
                       horizontal_linecuts=None,
                       x_lims=False,
                       plot = True):
    # loading data file for the first time
    data = np.loadtxt(inputFile, skiprows=1)
    x = conversion_x * np.array(data[:, col_x])  # fast scan direction (usually second column)
    y = conversion_y * np.array(data[:, col_y])  # slow scan direction (usually first column)
    z = conversion_z * np.array(data[:, col_z])  # to plot
    idx = np.nonzero(x == x[0])[0]
    len_x = idx[1]

    # looping until x, y, and z cam be reshaped correctly.
    while (len(x) % len_x != 0):
        time.sleep(0.3)
        data = np.loadtxt(inputFile, skiprows=1)
        x = conversion_x * np.array(data[:, col_x])  # fast scan direction (usually second column)
        y = conversion_y * np.array(data[:, col_y])  # slow scan direction (usually first column)
        z = conversion_z * np.array(data[:, col_z])  # to plot
        idx = np.nonzero(x == x[0])[0]
        len_x = idx[1]

    x = x.reshape((-1, len_x))
    y = y.reshape((-1, len_x))
    z = z.reshape((-1, len_x))
    flatx = x[0, :]
    flaty = y[:, 0]

    if plot is True:
        fig, ax1 = plt.subplots(figsize=(4, 4), nrows=1, ncols=1, squeeze=True)
        ax1.set_xlabel(label_x)
        ax1.set_ylabel(label_z)

    if horizontal_linecuts is not None:
        for y_cut in horizontal_linecuts:
            y_cut_real = find_nearest(flaty, y_cut)
            data_hcut = data[data[:, col_y] == y_cut_real]
            z_cut = conversion_z * np.array(data_hcut[:, col_z])
            x_cut = conversion_x * np.array(data_hcut[:, col_x])
            if plot is True:
                ax1.plot(x_cut, z_cut, label=str(np.round(y_cut_real, 2)))
                ax1.legend(loc=0)
        if x_lims is not False:
            if plot is True:
                ax1.set_xlim(x_lims[0], x_lims[1])

    else:
        if plot is True:
            ax1.remove()
    return x_cut, z_cut


#function for extracting and plotting a vertical linecut of a 2D data plot
#this function does not work super well if your conversion_x is a rational number. Keep it 1.
def vertical_linecut(inputFile,
                       col_x, label_x, conversion_x,
                       col_y, label_y, conversion_y,
                       col_z, label_z, conversion_z,
                       vertical_linecuts=None,
                       y_lims = False,
                       plot = True):
    # loading data file for the first time
    data = np.loadtxt(inputFile, skiprows=1)
    x = conversion_x * np.array(data[:, col_x])  # fast scan direction (usually second column)
    y = conversion_y * np.array(data[:, col_y])  # slow scan direction (usually first column)
    z = conversion_z * np.array(data[:, col_z])  # to plot
    idx = np.nonzero(x == x[0])[0]
    len_x = idx[1]

    # looping until x, y, and z cam be reshaped correctly.
    while (len(x) % len_x != 0):
        time.sleep(0.3)
        data = np.loadtxt(inputFile, skiprows=1)
        x = conversion_x * np.array(data[:, col_x])  # fast scan direction (usually second column)
        y = conversion_y * np.array(data[:, col_y])  # slow scan direction (usually first column)
        z = conversion_z * np.array(data[:, col_z])  # to plot
        idx = np.nonzero(x == x[0])[0]
        len_x = idx[1]

    x = x.reshape((-1, len_x))
    y = y.reshape((-1, len_x))
    z = z.reshape((-1, len_x))
    flatx = x[0, :]
    flaty = y[:, 0]
    z_cuts = []
    if plot is True:
        fig, ax1 = plt.subplots(figsize=(4, 4), nrows=1, ncols=1, squeeze=True)
        ax1.set_xlabel(label_y)
        ax1.set_ylabel(label_z)
        if vertical_linecuts is not None:
            ax1.set_xlabel(label_y)
            ax1.set_ylabel(label_z)
            for x_cut in vertical_linecuts:
                x_cut_index = find_nearest_index(flatx, x_cut)
                z_cut = z[:, x_cut_index]
                z_cuts.append(z_cut)
                ax1.plot(flaty, z_cut, label=str(np.round(flatx[x_cut_index], 3)))
            if y_lims is not False:
                ax1.set_xlim(y_lims[0], y_lims[1])
            ax1.legend(loc=0)
        else:
            ax1.remove()
        fig.tight_layout()

    if plot is False:
        for x_cut in vertical_linecuts:
            x_cut_index = find_nearest_index(flatx, x_cut)
            z_cut = z[:, x_cut_index]
            z_cuts.append(z_cut)
    return flaty, z_cuts




def magnetoconductance_range_special_bandpass(inputFiles, column_B, column_G, R_lines, V_AC, windows, orders, labels,
                                              plot_name, A_min, A_center, A_max, B_min, B_max, order_filter, fs, highpass_cutoff, lowpass_cutoff,
                                              smooth_window, f_max = 2000, angle_correction = True, smooth_data = True, plot=True, fitting=True, bandpass=False,
                                              smooth_fft=True, locked_fit=True, only_fft=True):
    # A min and A max are given in um2, they are used to calculate the expected limit of AB
    # B_min and B_max indicate the range of the data points in which compute the FFT of the data
    # Rlines in Ohm
    # V_AC in voltage

    B_1 = 4.135667513e-15 * 1e12 / A_min
    B_center = 4.135667513e-15 * 1e12 / A_center
    B_2 = 4.135667513e-15 * 1e12 / A_max
    f_1 = 1.0 / B_1
    f_center = 1. / B_center
    f_2 = 1.0 / B_2

    x1s = []
    A1s = []
    B1s = []
    x2s = []
    A2s = []
    B2s = []
    offsets = []

    x1errs = []
    A1errs = []
    B1errs = []
    x2errs = []
    A2errs = []
    B2errs = []
    offseterrs = []

    freq_all = []
    fft_all = []

    for inputFile in inputFiles:

        n = inputFiles.index(inputFile)
        data = []
        B = []
        G = []
        B_interp = []
        G_interp = []
        diff = []
        freq_fft = []
        residuals_fft = []
        freq_fft_fit_AB = []
        amplitude_fft_fit_AB = []
        freq_fft_fit = []
        amplitude_fft_fit = []

        shift_ax2 = 0.02
        shift_ax3 = 3

        data = np.loadtxt(inputFile)
        B = np.array(data[1:, column_B])

        if angle_correction is True:
            phi_offset_deg = 90 + 9.4
            phi_offset_rad = phi_offset_deg * np.pi / 180.0
            B = B / np.sin(phi_offset_rad)


        lockin_x = np.array(data[1:, column_G])  # Lx in Volt for sag data
        Imeas = 1e6
        G = 12906 * (lockin_x / Imeas) / (V_AC - lockin_x / Imeas * R_lines)

        #smooth conductance
        if smooth_data is True:
            G = sci.signal.savgol_filter(G, 11, 3)

        # Bandpass filter:
        if bandpass is True:
            # Filter the data, and plot both the original and filtered signals.
            G_filter = butter_bandpass_filter(G, highpass_cutoff, lowpass_cutoff, fs, order_filter)

            B = B[50:]
            G_filter = G_filter[50:]

        else:
            G_filter = G

        # flipping in  case of backward scan in field
        if B[0] > B[-1]:
            B = np.flipud(B)
            G_filter = np.flipud(G_filter)

        B_interp = interpolation(B, G_filter)[0]
        G_interp_filter = interpolation(B, G_filter)[1]

        B_interp, G_interp_filter = cutter(B_interp, G_interp_filter, B_min, B_max)

        B_step = np.abs(B_interp[0] - B_interp[-1]) / len(B_interp)
        window = windows[inputFiles.index(inputFile)]
        order = orders[inputFiles.index(inputFile)]

        diff, G_smooth_filter = residuals(G_interp_filter, window, order)

        residuals_fft = np.fft.fft(diff)  # is a complex number
        amplitude_fft = np.absolute(residuals_fft)
        freq_fft = np.fft.fftfreq(len(B_interp), d=B_step)

        if smooth_fft is True:
            # smooth FFT
            amplitude_fft = savgol_filter(amplitude_fft, smooth_window, 3)

        x1 = f_center
        x2 = 2 * f_center

        if fitting is True:

            x1 = f_center
            x2 = 2 * f_center
            A1 = 1
            B1 = 10
            A2 = 0.2
            B2 = 15
            offset = 0

            if locked_fit == True:
                x1, A1, B1, x2, A2, B2, offset, x1err, A1err, B1err, x2err, A2err, B2err, offseterr = \
                    fitting_special_function_twopeaks_locked(freq_fft, amplitude_fft, limit_fit, x1, A1, B1, A2, B2,
                                                             offset)

            else:
                x1, A1, B1, x2, A2, B2, offset, x1err, A1err, B1err, x2err, A2err, B2err, offseterr = \
                    fitting_special_function_twopeaks(freq_fft, amplitude_fft, limit_fit, x1, A1, B1, x2, A2, B2,
                                                      offset)
                # x1, A1, B1, offset, x1err, A1err, B1err, offseterr = \
                # fitting_special_function(freq_fft, amplitude_fft, limit_fit, x1, A1, B1, offset)

            freq = np.linspace(2, limit_fit, 2000)
            fit = special_function_twopeaks(freq, x1, A1, B1, 2 * x1, A2, B2, offset)
            # fit = special_function(freq, x1, A1, B1, offset)
            x1s.append(x1)
            A1s.append(A1)
            B1s.append(B1)
            x2s.append(x2)
            A2s.append(A2)
            B2s.append(B2)
            offsets.append(offset)

            x1errs.append(x1err)
            A1errs.append(A1err)
            B1errs.append(B1err)
            x2errs.append(x2err)
            A2errs.append(A2err)
            B2errs.append(B2err)
            offseterrs.append(offseterr)


            # this is a way to cut only some part of the FFT

        if fitting is False:
            x1s.append(0)
            A1s.append(0)
            B1s.append(0)
            x2s.append(0)
            A2s.append(0)
            B2s.append(0)
            offsets.append(0)

            x1errs.append(0)
            A1errs.append(0)
            B1errs.append(0)
            x2errs.append(0)
            A2errs.append(0)
            B2errs.append(0)
            offseterrs.append(0)

        freq_fft_fit, amplitude_fft_fit = cutter(freq_fft, amplitude_fft, 0, 16000)

        freq_all.append(freq_fft_fit)
        fft_all.append(amplitude_fft_fit)

        if plot is True:

            if only_fft == True:
                fig = plt.figure(figsize=(11, 20))
                ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=2, rowspan=1)
                ax3.set_xlabel('$ B^{-1} (T^{-1}) $', fontsize=axes_font)
                ax3.set_ylabel('Amplitude', fontsize=axes_font)
                ax3.tick_params(axis='both', which='major', labelsize=tick_font)
                ax3.get_xaxis().get_major_formatter().set_useOffset(False)

                ax3.plot(freq_fft_fit, amplitude_fft_fit + shift_ax3 * n, linewidth=1.5, label=labels[n], color='black')
                ax3.plot(freq, fit + shift_ax3 * n, color='orange', linewidth=2.5)
                ax3.axvspan(f_1, f_2, facecolor='xkcd:lavender', alpha=0.3)
                ax3.axvline(x1, linestyle='--', color='k', label='AB from fit')
                ax3.axvline(x2, linestyle='--', color='k', label='AAS from fit')

                ax3.legend(loc=0)

            else:
                fig = plt.figure(figsize=(8, 12))
                ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
                ax2 = plt.subplot2grid((4, 2), (1, 0), colspan=2)
                ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=2, rowspan=1)

                # ax1.grid()
                #ax1.set_xlabel('$ B (T) $', fontsize=axes_font)
                #ax1.set_ylabel('G ($ 2e^2/h$)', fontsize=axes_font)
                ax1.tick_params(axis='both', which='major', labelsize=tick_font)
                ax1.get_xaxis().get_major_formatter().set_useOffset(False)
                ax1.set_xlim(B_min, B_max)
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])



                # ax2.grid()
                #ax2.set_xlabel(' $ B (T) $', fontsize=axes_font)
                #ax2.set_ylabel('Residuals G ($ 2e^2/h$)', fontsize=axes_font)
                ax2.tick_params(axis='both', which='major', labelsize=tick_font)
                ax2.get_xaxis().get_major_formatter().set_useOffset(False)
                ax2.set_xlim(B_min, B_max)
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])

                # ax3.grid()
                #ax3.set_xlabel('$ B^{-1} (T^{-1}) $', fontsize=axes_font)
                #ax3.set_ylabel('Amplitude', fontsize=axes_font)
                ax3.tick_params(axis='both', which='major', labelsize=tick_font)
                ax3.get_xaxis().get_major_formatter().set_useOffset(False)
                ax3.set_xlim(0, f_max)
                ax3.grid(False)
                ax3.set_xticklabels([])
                ax3.set_yticklabels([])

                ax1.plot(B_interp, G_interp_filter, linewidth=1.5, label=labels[n], color='black')
                ax1.plot(B_interp, G_smooth_filter, linewidth=1.5, label='smooth', color='orange')
                ax2.plot(B_interp, diff + shift_ax2 * n, linewidth=1.5, label=labels[n], color='black')
                ax3.plot(freq_fft_fit, amplitude_fft_fit + shift_ax3 * n, linewidth=1.5, label=labels[n], color='black')
                if fitting is True:
                    ax3.plot(freq, fit + shift_ax3 * n, color='orange', linewidth=2.5)
                # ax3.axvline( x1, linestyle = '--', color = 'k', label='AB from fit')
                # ax3.axvline( x2, linestyle = '--', color = 'k', label='AAS from fit')
                ax3.axvspan(f_1, f_2, facecolor='xkcd:lavender', alpha=0.3)

                # xkcd:chartreuse
                # xkcd:aqua
                # xkcd:lime
                # darkturquoise

                #ax1.legend(loc=0)
                #ax2.legend(loc=0)
                #ax3.legend(loc=0)

                fig.subplots_adjust(hspace=sub_hspace)
                fig.subplots_adjust(wspace=sub_wspace)
                fig.tight_layout()
                fig.savefig('magnetoconductance' + str(plot_name) + '.eps', format='eps', transparent=False)
                fig.savefig('magnetoconductance' + str(plot_name) + '.png', format='png', transparent=False)

    return freq_all, fft_all, x1s, A1s, B1s, x2s, A2s, B2s, x1errs, A1errs, B1errs, x2errs, A2errs, B2errs



#function that return residuals in a 2D data plot as a function of a 'gate' parameter
def residuals_2D(inputFile, column_B, column_G, window, order,  B_min, B_max, hyst_offset = None):

    data = np.loadtxt(inputFile, skiprows=1)
    B = np.array(data[:, column_B])
    # if column B refers to By and this measurement is taken in B2, consider the offset of phi
    phi_offset_deg = 90 + 9.4
    phi_offset_rad = phi_offset_deg * np.pi / 180.0
    B = B / np.sin(phi_offset_rad)
    G = np.array(data[:, column_G])
    G = sci.signal.savgol_filter(G, 7, 2)
    # flipping in  case of backward scan in field
    if B[0] > B[-1]:
        B = np.flipud(B)
        G = np.flipud(G)
    B_interp = interpolation(B, G)[0]
    G_interp = interpolation(B, G)[1]

    # adding hyst offset for down sweeps
    if hyst_offset is not None:
        B_interp, G_interp = cutter(B_interp - hyst_offset, G_interp, B_min, B_max)
    else:
        B_interp, G_interp = cutter(B_interp, G_interp, B_min, B_max)

    diff, G_smooth = residuals(G_interp, window, order)

    #Smoothing the diff
    diff = sci.signal.savgol_filter(diff, 21, 5)

    return B_interp, G_interp, diff, G_smooth


def colorplot_residuals(inputFiles_up, inputFiles_down, column_B, column_G, window, order, B_min, B_max, gates, hyst_offset):

    B_interps = []
    G_interps = []
    diffs = []
    G_smooths = []
    ns = []

    if inputFiles_down is not None:
        for (inputFile_up, inputFile_down) in zip(inputFiles_up, inputFiles_down):
            out_up = residuals_2D(inputFile_up, column_B, column_G, window, order,  B_min, B_max);
            ns.append(len(out_up[0]))
            B_interps.append(out_up[0])
            G_interps.append(out_up[1])
            diffs.append(out_up[2])
            G_smooths.append(out_up[3])
            out_down = residuals_2D(inputFile_down, column_B, column_G, window, order, B_min, B_max, hyst_offset = hyst_offset);
            ns.append(len(out_down[0]))
            B_interps.append(out_down[0])
            G_interps.append(out_down[1])
            diffs.append(out_down[2])
            G_smooths.append(out_down[3])
        n_min = np.min(ns)
        Bs = []
        residuals =[]
        for B_interp, diff in zip(B_interps, diffs):
            Bs.append(interpol(B_interp, diff, n_min)[0])
            residuals.append(interpol(B_interp, diff, n_min)[1])
        flat_B = Bs[0]
        flat_gate = gates
        residuals = np.array(residuals)
        Residuals = np.reshape(residuals, [len(flat_gate), len(flat_B),])

        fig2D = plt.figure(figsize =(5,5))

        if (flat_gate[0] < flat_gate[-1]):
            im = plt.imshow((Residuals)
                            , cmap=plt.cm.seismic
                            , interpolation='none'
                            , origin='lower'
                            , extent=[flat_B[1],flat_B[-1],flat_gate[0], flat_gate[-1]]
                            , aspect="auto")
        else:
            im = plt.pcolor( flat_B, flat_gate, Residuals
                             , cmap = plt.cm.seismic
                             )
        plt.ylim(min(flat_gate), max(flat_gate))
        plt.xlabel(' $ B _{\perp} $ (T)')
        plt.ylabel( ' $E_{QD}$ (mV) ')
        cbar = plt.colorbar(im, orientation='vertical')
        cbar.set_label('  | $ \delta $ G ($2e^2/h $)| ')

    else:
        for inputFile_up in inputFiles_up:
            out_up = residuals_2D(inputFile_up, column_B, column_G, window, order,  B_min, B_max);
            ns.append(len(out_up[0]))
            B_interps.append(out_up[0])
            G_interps.append(out_up[1])
            diffs.append(out_up[2])
            G_smooths.append(out_up[3])
        n_min = np.min(ns)
        Bs = []
        residuals =[]
        for B_interp, diff in zip(B_interps, diffs):
            Bs.append(interpol(B_interp, diff, n_min)[0])
            residuals.append(interpol(B_interp, diff, n_min)[1])
        flat_B = Bs[0]
        flat_gate = gates
        residuals = np.array(residuals)
        Residuals = np.reshape(residuals, [len(flat_gate), len(flat_B),])
        fig2D = plt.figure(figsize =(6,6))
        if (flat_gate[0] < flat_gate[-1]):
            im = plt.imshow((Residuals), cmap=plt.cm.seismic, interpolation='none', origin='lower', extent=[flat_B[1],flat_B[-1],flat_gate[0], flat_gate[-1]],
                            aspect="auto")
        else:
            im = plt.pcolor(flat_B, flat_gate, Residuals
                            , cmap=plt.cm.seismic)
        plt.ylim(min(flat_gate), max(flat_gate))
        plt.xlabel(' $ B _{\perp} $ (T)')
        plt.ylabel( ' $E_{QD}$ ')
        cbar = plt.colorbar(im, orientation='vertical')
        cbar.set_label('  G ($2e^2/h $) ')

    #new part that plots a waterfall of fitted traces with detection of maxima.

    # guess parameters for fitting of single sinusoidal
    A_0, B_0, C_0, D_0 = 0.015, 300, 2, 0
    parameters = []
    y_fits = []
    first_index = 0
    last_index = len(gates)

    for i in range(0, last_index):
        params, params_covariance = op.curve_fit(fit_sinusoidal, flat_B, Residuals[i, :], p0=[A_0, B_0, C_0, D_0])
        y_fit = fit_sinusoidal(flat_B, params[0], params[1], params[2], params[3])
        parameters.append(params)
        y_fits.append(y_fit)

    plt.figure(figsize=(4, 18))
    off = 0.06
    plt.xlabel('$ B_{perp} $ (T)', fontsize=axes_font)
    plt.ylabel(' G', fontsize=axes_font)

    for i, par in zip(range(0, last_index), parameters):
        peakind = signal.find_peaks_cwt(y_fits[i], np.arange(1, 10))
        plt.plot(flat_B[peakind], y_fits[0][peakind] + off * i, 'ro', color='red', markersize=7)
        plt.plot(flat_B, Residuals[i, :] + off * i, linestyle='--', color='black')
        plt.plot(flat_B, y_fits[i] + off * i, color='black')

        #print(flat_B[peakind][0])
        #print(flat_B[peakind][1])


    parameters = np.array(parameters)
    y_fits = np.array(y_fits)
    parameters = parameters.reshape((-1, len(params)))

    return flat_B, flat_gate, Residuals


def pi_shift_collection(inputFiles_up, inputFiles_down,
                        gate_inputs,
                        gate_range_up,
                        gate_range_down,
                        offset=0,
                        hyst_offset=-0.003):
    # single direction for now
    # adding the second direction
    # gate_inputs is the list of file of gate sweep
    # gate_range is the array of gates
    starting_points_up = []
    starting_points_down = []

    fig = plt.figure(figsize=(7, 24))

    plt.ylabel('G $ (2e^2/h $)', fontsize=axes_font)
    plt.xlabel(' $ B_{\perp} (T) $', fontsize=axes_font)

    for magnetoconductance_input_up in inputFiles_up:
        index = inputFiles_up.index(magnetoconductance_input_up)
        B, G = extract(magnetoconductance_input_up, 2, 8)[0], extract(magnetoconductance_input_up, 2, 8)[1]
        starting_points_up.append(G[0])
        if gate_range_up is not None:
            plt.plot(B, G + offset * index, linewidth=2, label=str(gate_range_up[index]))
        else:
            plt.plot(B, G + offset * index, linewidth=2, label=str(inputFiles_up[index]))

    if inputFiles_down is not None:
        for magnetoconductance_input_down in inputFiles_down:
            index = inputFiles_down.index(magnetoconductance_input_down)
            B, G = extract(magnetoconductance_input_down, 2, 8)[0], extract(magnetoconductance_input_down, 2, 8)[1]
            starting_points_down.append(G[-10])
            if gate_range_up is not None:
                plt.plot(B - hyst_offset, G + offset * (index + 0.5), linewidth=2, label=str(gate_range_down[index]))
            else:
                plt.plot(B - hyst_offset, G + offset * (index + 0.5), linewidth=2,
                         label=str(magnetoconductance_input_down[index]))

    plt.grid(True)

    # treating gates data
    fig = plt.figure(figsize=(5, 4))
    plt.ylabel('G $ (2e^2/h $)')
    plt.xlabel('$E_{QD}$ ($ meV $)')

    if gate_inputs is not None:
        for gate_input in gate_inputs:
            gate, G = extract(gate_input, 0, 5)[0], extract(gate_input, 0, 5)[1]
            plt.plot(gate, G, linewidth=2, label=str(gate_input))
        plt.legend(loc=0)
    if gate_range_up is not None:
        plt.scatter(gate_range_up, starting_points_up, marker='o', color='black', label='up')
    if gate_range_down is not None:
        plt.scatter(gate_range_down, starting_points_down, marker='o', color='black', label='down')
    plt.legend(loc=0)
    plt.tight_layout()

    return  starting_points_up,  starting_points_down


# for a single resonance
def S(J_l, J_r, e_r, x, phi, t_ref, ka, e_d):
    eps = -2*np.cos(ka)
    S = t_ref + (J_l*np.exp(1j*phi*(1.0+x)) * np.conjugate(J_r))/(eps - e_d + e_r)
    return S



#Jref = jr*j.
def T(e_d, J_l, J_r, e_r, x, phi, t_ref, ka, offset):
    eps = -2*np.cos(ka)
    S_lr = S(J_l, J_r, e_r, x, phi, t_ref, ka, e_d)
    S_ll = S(J_l, J_l, e_r, x, 0, t_ref, ka, e_d)
    S_rr = S(J_r, J_r, e_r, x, 0, t_ref, ka, e_d)
    T = (offset + 4*math.sin(ka)**2*(np.abs(S_lr))**2 / (np.abs( (np.abs(S_lr))**2 - (S_ll + np.exp(-1j*ka))*(S_rr + np.exp(-1j*ka))))**2)
    return T

def S_2(J_l1, J_r1, e_r1, x1,
        J_l2, J_r2, e_r2, x2,
        phi,
        t_ref, ka, e_d):
    eps = -2*np.cos(ka)
    S = t_ref + \
        (J_l1*np.exp(1j*phi*(1.0+x1)) * np.conjugate(J_r1))/(eps - e_d + e_r1) + \
        (J_l2*np.exp(1j*phi*(1.0+x2)) * np.conjugate(J_r2))/(eps - e_d + e_r2)
    return S

def T_2(e_d,
        J_l1, J_r1, e_r1, x1,
        J_l2, J_r2, e_r2, x2,
        phi, t_ref, ka, offset):
    eps = -2*np.cos(ka)
    S_lr = S_2(J_l1, J_r1, e_r1, x1,
                J_l2, J_r2, e_r2, x2,
                phi, t_ref, ka, e_d)
    S_ll = S_2(J_l1, J_l1, e_r1, x1,
                J_l2, J_l2, e_r2, x2,
                0.0, t_ref, ka, e_d)
    S_rr = S_2(J_r1, J_r1, e_r1, x1,
                J_r2, J_r2, e_r2, x2,
                0.0, t_ref, ka, e_d)
    T_2 = (offset + 4*math.sin(ka)**2*(np.abs(S_lr))**2 / (np.abs( (np.abs(S_lr))**2 - (S_ll + np.exp(-1j*ka))*(S_rr + np.exp(-1j*ka))))**2)
    return T_2

def S_4(J_l1, J_r1, e_r1, x1,
        J_l2, J_r2, e_r2, x2,
        J_l3, J_r3, e_r3, x3,
        J_l4, J_r4, e_r4, x4,
        phi,
        t_ref, ka, e_d):
    eps = -2*np.cos(ka)
    S = t_ref + \
        (J_l1*np.exp(1j*phi*(1.0+x1)) * np.conjugate(J_r1))/(eps - e_d + e_r1) + \
        (J_l2*np.exp(1j*phi*(1.0+x2)) * np.conjugate(J_r2))/(eps - e_d + e_r2) + \
        (J_l3*np.exp(1j*phi*(1.0+x3)) * np.conjugate(J_r3))/(eps - e_d + e_r3) + \
        (J_l4*np.exp(1j*phi*(1.0+x4)) * np.conjugate(J_r4))/(eps - e_d + e_r4)
    return S

def T_4(e_d,
        J_l1, J_r1, e_r1, x1,
        J_l2, J_r2, e_r2, x2,
        J_l3, J_r3, e_r3, x3,
        J_l4, J_r4, e_r4, x4,
        phi, t_ref, ka, offset):
    eps = -2*np.cos(ka)
    S_lr = S_4(J_l1, J_r1, e_r1, x1,
                J_l2, J_r2, e_r2, x2,
                J_l3, J_r3, e_r3, x3,
                J_l4, J_r4, e_r4, x4,
                phi, t_ref, ka, e_d)
    S_ll = S_4(J_l1, J_l1, e_r1, x1,
                J_l2, J_l2, e_r2, x2,
                J_l3, J_l3, e_r3, x3,
                J_l4, J_l4, e_r4, x4,
                0.0, t_ref, ka, e_d)
    S_rr = S_4(J_r1, J_r1, e_r1, x1,
                J_r2, J_r2, e_r2, x2,
                J_r3, J_r3, e_r3, x3,
                J_r4, J_r4, e_r4, x4,
               0.0, t_ref, ka, e_d)
    T_4 = (offset + 4*math.sin(ka)**2*(np.abs(S_lr))**2 / (np.abs( (np.abs(S_lr))**2 - (S_ll + np.exp(-1j*ka))*(S_rr + np.exp(-1j*ka))))**2)
    return T_4
