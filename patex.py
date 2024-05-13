# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:34:03 2023

@author: Zach Drinkwater
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy

def idx_to_angle(idx):
    return idx/100 + 15

#Straight line
def line(x, m, c):
    return m * x + c

#Gaussian curve
def gaussian(x, amp, cen, sigma):
    return amp * (1 / (sigma * (np.sqrt(2*np.pi)))) * (np.exp((-1.0/2.0) * (((x-cen)/sigma)**2)))

# Cauchy-Lorentz curve
def cauchy_lorentz(x, amp, cen, gamma):
    return amp * 1 / (np.pi * gamma * (1 + ((x - cen)/gamma) ** 2) )

# https://docs.mantidproject.org/nightly/fitting/fitfunctions/PseudoVoigt.html

# Pseudo-Voigt curve - suggest not to use
def pseudo_voigt(x, amp, cen, sigma, gamma, eta):
    #eta = 0.9
    return eta * gaussian(x, amp, cen, sigma) + (1 - eta) *  cauchy_lorentz(x, amp, cen, gamma)

# Special W function for Voigt
def w(z):
    return scipy.special.wofz(z)

# Voigt curve
def voigt(x, amp, cen, sigma, gamma):
    z = (x - cen + gamma * 1j)/(sigma * np.sqrt(2))
    return amp * np.real(w(z))/(sigma * np.sqrt(2 * np.pi))

class XRDPeakData:
    """
    XRDPeakData objects can be created from XRD datasets and provide a quick
    way to visualise and fit data.
    """
    def __init__(self, filename, lam, skiprows):
        """
        Parameters
        ----------
        filename : string
            Filename of the dataset (txt file).
        lam : float
            The X-ray wavelength used in the XRD experiment.
        """
        self.data = np.loadtxt(filename, skiprows=skiprows)
        self.lam = lam
        self.two_theta = self.data[:, 0]
        self.intensity = self.data[:, 1]
        self.lattice_parameter = 0
        self.lattice_parameter_err = 0

        
    def filter_data(self, sigma):
        """
        Filters self.intensity data using a 1-D Gaussian filter.

        Parameters
        ----------
        sigma : integer
            Order of Gaussian filter.

        Returns
        -------
        None.

        """
        self.intensity = scipy.ndimage.gaussian_filter1d(self.intensity, sigma)
        
    # def scipy_peak_data(self, min_peak_intensity, max_peak_intensity, idx_halfrange, distance, prominence):
    #     peaks = scipy.signal.find_peaks(self.intensity, height=[min_peak_intensity, max_peak_intensity], distance=distance, prominence=prominence)
    #     return peaks
    
    def plot_raw_peak_data(self, log=True):
        """
        Plots the raw data from the data file, with the option to crop it.

        Parameters
        ----------
        log : Bool
            Whether to take log base 10 of intensity data. The default is False.

        Returns
        -------
        None.

        """
        new_int = self.intensity

        
        fig = plt.figure(figsize=(11, 3))
        ax = fig.add_subplot(111)
        if log == True:
            ax.plot(self.two_theta, np.log10(new_int + 1))
        else:
            ax.plot(self.two_theta, new_int)
        plt.xlabel(r'$2\theta$')
        plt.ylabel('log(intensity)')
        plt.show()
        
        # ax = plt.axes()
        # ax.plot(self.two_theta, np.log10(new_int + 1))
       
        
    def cut_range(self, idx, idx_halfrange):
        """
        Cuts the 2-theta and intesnity data down to a smaller range.

        Parameters
        ----------
        idx : int
            The centre of the range.
        idx_halfrange : int
            The half-width of the range.

        Returns
        -------
        theta_range : np.array
            2-theta range of interest.
        intensity_range : np.array
            Intensity data range of interest.

        """
        u_idx = idx + idx_halfrange
        l_ldx = idx - idx_halfrange
        theta_range = self.two_theta[l_ldx:u_idx]
        intensity_range = self.intensity[l_ldx:u_idx]
        return theta_range, intensity_range
    
    def plot_local_data(self, idx, idx_halfrange, crop=False):
        """
        Plots the data around idx with half-width idx_halfrange.

        Parameters
        ----------
        idx : int
            The centre of the range.
        idx_halfrange : int
            The half-width of the range.
        crop : Bool, optional
            Whether or not to crop the data. The default is False.

        Returns
        -------
        None.

        """
        ax = plt.axes()
        if crop == True:
            theta_range, intensity_range = self.cut_range(idx, idx_halfrange)
            new_int = intensity_range
            for i in range(len(new_int)):
                if new_int[i] > 500:
                    new_int[i] = 500    
            ax.plot(theta_range, new_int, '.', markersize=5, label="Data")
        else:
            theta_range, intensity_range = self.cut_range(idx, idx_halfrange)
            ax.plot(theta_range, intensity_range, '.', markersize=5, label="Data")
            
        ax.set_ylabel('Intensity (1e6)')
        ax.set_xlabel(r'$2\theta$')
        plt.legend()
        plt.show()

    def plot_peak_data(self, min_peak_intensity, max_peak_intensity, idx_halfrange, distance, prominence, crop=False):
        """
        Plots the raw data around the peaks found by SciPy.

        Parameters
        ----------
        Peak parameters (various)
        crop : bool, optional
            Whether or not to crop the data. The default is False.

        Returns
        -------
        None.

        """
        peaks = scipy.signal.find_peaks(self.intensity, height=[min_peak_intensity, max_peak_intensity], distance=distance, prominence=prominence)
        print(peaks[0])
        for idx in peaks[0]:
            self.plot_local_data(idx, idx_halfrange, crop=crop)
            
    def fit_all_peaks(self, peak_params, curve='g', film=True, film_left=True, num_peaks=4, plot=True, Q=False):
        """
        

        Parameters
        ----------
        peak_params : list 
            A list of parameters for confining which peaks to identify.
        curve : string, optional
            Which type of curve to use. 'g' is Gaussian, 'l' is Cauchy-Lorentz,
            and 'v' is Voigt. The default is 'g'.
        film : Bool, optional
            Whether to fit film or substrate peaks. The default is True (film).
        film_left : Bool, optional
            Whether the peaks are on the left (True) or right (False) of the
            substrate peaks. The default is True.
        num_peaks : int, optional
            The expected number of substrate OR film peaks. The default is 4.
        plot : Bool, optional
            Whether to output a plot of the fitted curves. The default is True.
        Q : Bool, optional
            If True, then the fitting is done in reciprocal lattice units. 
            Otherwise, the fitting is done in degrees. The default is False.

        Returns
        -------
        peak_data: list
            An array containing peak information for all fitted peaks.

        """
        
        
        min_peak_intensity, max_peak_intensity, idx_halfrange, distance, prominence, width = peak_params
        
        # Find the peaks
        peaks = scipy.signal.find_peaks(self.intensity, height=[min_peak_intensity, max_peak_intensity], distance=distance, prominence=prominence, width=width)
        #print(sig.peak_prominences(self.intensity, peaks[0], wlen=None)[0])
        
        # Ordering the peaks by peak height
        Z = [x for _, x in sorted(zip(peaks[1]['peak_heights'], peaks[0]))]
        ordered_peak_idices = np.flip(np.array(Z))
        
        # Separating film and substrate peak locations
        # (Assuming four largest peaks correspond to substrate peaks)
        film_peaks = np.sort(ordered_peak_idices[4:])
        substrate_peaks = np.sort(ordered_peak_idices[0:num_peaks])
        
        # Creating desired peaks list depending on user's choice.
        # The desired_peaks list includes the indices of peaks of either the 
        # film or substrate peaks.
        desired_peaks = []
        if film == True:
            if film_left == True:
                print("Film peaks found at", idx_to_angle(film_peaks)) # CONVERT TO DEGREES IN ERROR MESSAGE
                #print("Substrate peaks found at", idx_to_angle(substrate_peaks))
                
                try:
                    desired_peaks = [film_peaks[i] for i in range(len(substrate_peaks)) if film_peaks[i] < substrate_peaks[i]]
                except IndexError:
                    print(f'REQUIRES AT LEAST {num_peaks} FILM PEAKS, FOUND {len(film_peaks)}')
                    return 0
            else:
                print("Film peaks found at", idx_to_angle(film_peaks)) # CONVERT TO DEGREES IN ERROR MESSAGE
                print("Substrate peaks found at", idx_to_angle(substrate_peaks))
                
                try:
                    desired_peaks = [film_peaks[i] for i in range(len(substrate_peaks)) if film_peaks[i] > substrate_peaks[i]]
                except IndexError:
                    print(f'REQUIRES AT LEAST {num_peaks} FILM PEAKS, FOUND {len(film_peaks)}')
                    return 0
        else:
            print("Substrate peaks found at", idx_to_angle(substrate_peaks))
            desired_peaks = substrate_peaks
        
        # Iterating through the desired peaks.
        i = 0
        #print(desired_peaks)
        peak_data = np.ones((6, np.size(desired_peaks)))
        for idx in desired_peaks:
            #Initial parameter guesses
            amp = self.intensity[idx]
            cen = self.two_theta[idx]
            sigma = 0.05
            gamma = 0.05
            eta = 0.1
            
            # Range (in 2theta) over which to fit
            theta_range, intensity_range = self.cut_range(idx, idx_halfrange)
            
            if Q == True: # For Williamson-Hall, need to use Q instead of 2 theta
                theta_range = np.sin((np.pi/180) * theta_range/2)
                cen = np.sin((np.pi/180) * cen/2)
                sigma = np.sin((np.pi/180) * sigma/2)
                
            
            # Fitting the peaks based on user's curve choice.
            if curve == 'g':
                p0 = [amp, cen, sigma]
                peak_data[:, i] = self.fit_gaussian(theta_range, intensity_range, p0, plot=plot)
            elif curve == 'l':
                p0 = [amp, cen, gamma]
                peak_data[:, i] = self.fit_lorentz(theta_range, intensity_range, p0, plot=plot)
            elif curve == 'v':
                p0 = [amp, cen, sigma, gamma]
                peak_data[:, i] = self.fit_voigt(theta_range, intensity_range, p0, plot=plot)
            elif curve == 'pv':
                p0 = [amp, cen, sigma, gamma, eta]
                peak_data[:, i] = self.fit_pseudo_voigt(theta_range, intensity_range, p0, plot=plot)
                
            i+=1
        self.lattice_parameter, self.lattice_parameter_err = self.get_lattice_parameter(peak_data, plot=plot)
        self.peak_data = peak_data
        
        return peak_data
        
    
    def fit_gaussian(self, theta_range, intensity_range, p0, return_params=False, plot=True):
        """
        Fits data to a Gaussian curve.

        Parameters
        ----------
        theta_range : NumPy Array
            Array of angles (or reciprocal lattice units) to fit over.
        intensity_range : Array of intensities from data.
            DESCRIPTION.
        p0 : list
            A list of initial parameter guesses.
        return_params : Bool, optional
            DESCRIPTION. The default is False.
        plot : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        popt, pcov = scipy.optimize.curve_fit(gaussian, theta_range, intensity_range, p0=p0, maxfev=1000000)
        perr = np.sqrt(np.diag(pcov))
        
        peak_amp = popt[0]
        peak_location = popt[1]
        peak_sigma = popt[2]
        FWHM = 2.35 * peak_sigma
        peak_intensity = peak_amp/(np.sqrt(2*np.pi)*peak_sigma)
        
        peak_amp_err = perr[0]
        #peak_loc_err = perr[1] # Using SciPy's curve fit error
        peak_loc_err = FWHM/2 # Using half of FWHM as error
        peak_sigma_err = perr[2]
        FWHM_err = 2.35 * peak_sigma_err
        peak_int_err = np.sqrt(1/(2*np.pi*peak_sigma**2)*peak_amp_err**2 + peak_amp**2/(2*np.pi*peak_sigma**4)*peak_sigma_err**2)
        
        if plot == True:
            ax = plt.axes()
            ax.plot(theta_range, intensity_range, '.', markersize=5, label="Data")
            x_array = np.linspace(min(theta_range), max(theta_range), 400)
            ax.plot(x_array, gaussian(x_array, *popt), label="Fitted Gaussian")
            ax.set_ylabel('Intensity')
            ax.set_xlabel(r'2$\theta$')
            plt.legend()
            plt.show()

        if return_params==True:
            return [peak_amp, peak_location, peak_sigma]
        else:
            return np.array([peak_location, peak_intensity, FWHM, peak_loc_err, peak_int_err, FWHM_err])
    
    def fit_lorentz(self, theta_range, intensity_range, p0, return_params=False, plot=True):
        
        popt, pcov = scipy.optimize.curve_fit(cauchy_lorentz, theta_range, intensity_range, p0=p0, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        peak_amp = popt[0]
        peak_location = popt[1]
        peak_gamma = popt[2]
        FWHM = 2 * peak_gamma
        peak_intensity = peak_amp/(np.pi * peak_gamma)
        
        peak_amp_err = perr[0]
        #peak_loc_err = perr[1] # Using SciPy's curve fit error
        peak_loc_err = FWHM/2 # Using half of FWHM as error
        peak_gamma_err = perr[2]
        FWHM_err = 2 * peak_gamma_err
        peak_int_err = np.sqrt(1/(np.pi**2 * peak_gamma**2) * peak_amp_err**2 + (peak_amp**2/peak_gamma**2) * peak_gamma_err**2)
        
        if plot == True:
            ax = plt.axes()
            ax.plot(theta_range, intensity_range, '.', markersize=5, label="Data")
            x_array = np.linspace(min(theta_range), max(theta_range), 400)
            ax.plot(x_array, cauchy_lorentz(x_array, *popt), label="Fitted Cauchy-Lorentz")
            ax.set_ylabel('Intensity')
            ax.set_xlabel(r'2$\theta$')
            plt.legend()
            plt.show()
                
        if return_params==True:
            return [peak_amp, peak_location, peak_gamma]
        else:
            return np.array([peak_location, peak_intensity, FWHM, peak_loc_err, peak_int_err, FWHM_err])
    
    def fit_voigt(self, theta_range, intensity_range, p0, return_params=False, plot=True):

        popt, pcov = scipy.optimize.curve_fit(voigt, theta_range, intensity_range, p0=p0, maxfev=100000)
        perr = np.sqrt(np.diag(pcov))
        peak_amp = popt[0]
        peak_location = popt[1]
        peak_sigma = popt[2]
        peak_gamma = popt[3]
        
        fg = 2.35 * peak_sigma
        fl = 2 * peak_gamma
        
        FWHM = 0.5346 * fl + np.sqrt(0.2166 * fl**2 + fg**2)
        peak_intensity = voigt(peak_location, peak_amp, peak_location, peak_sigma, peak_gamma)
        
        peak_amp_err = perr[0]
        #peak_loc_err = perr[1] # Using SciPy's curve fit error
        peak_loc_err = FWHM/2 # Using half of FWHM as error
        peak_sigma_err = perr[2]
        peak_gamma_err = perr[3]
        
        fg_err = 2.35 * peak_sigma_err
        fl_err = 2 * peak_gamma_err
        FWHM_err = np.sqrt(((0.2166 * fl)/(np.sqrt(0.2166 * fl**2 + fg**2)))**2 * fl_err**2 + ((fg)/(np.sqrt(0.2166 * fl**2 + fg**2))**2 * fg_err**2))
        
        dVdSigma = peak_amp / (peak_sigma**4 * np.sqrt(2*np.pi)) * (peak_gamma * peak_sigma * np.sqrt(2/np.pi) - (peak_gamma**2 + peak_sigma**2) * np.real(w(1j * peak_gamma / (peak_sigma * np.sqrt(2)))))
        dVdGamma = -peak_amp / (peak_sigma**3 * np.sqrt(2*np.pi)) * (peak_sigma * np.sqrt(2/np.pi) - peak_gamma * np.real(w(1j * peak_gamma / (peak_sigma * np.sqrt(2)))))
        dVdA = voigt(peak_location, peak_amp, peak_location, peak_sigma, peak_gamma) / peak_amp
        peak_int_err = np.sqrt(dVdSigma**2 * peak_sigma_err**2 + dVdGamma**2 * peak_gamma_err**2 + dVdA**2 * peak_amp_err**2)
        
        if plot == True:
            ax = plt.axes()
            ax.plot(theta_range, intensity_range, '.', markersize=5, label="Data")
            x_array = np.linspace(min(theta_range), max(theta_range), 400)
            ax.plot(x_array, voigt(x_array, *popt), label="Fitted Voigt")
            ax.set_ylabel('Intensity')
            ax.set_xlabel(r'2$\theta$')
            plt.legend()
            plt.show()      

        if return_params==True:
            return [peak_amp, peak_location, peak_sigma, peak_gamma]
        else:
            return np.array([peak_location, peak_intensity, FWHM, peak_loc_err, peak_int_err, FWHM_err])
    
    # def fit_pseudo_voigt(self, theta_range, intensity_range, p0, return_params=False, plot=True):
    #     ### WORK IN PROGRESS, DO NOT USE FOR NOW ###
    #     print('Psuedo-Voigt is a work in progress, it is highly suggested to use another curve.')
    #     popt, pcov = scipy.optimize.curve_fit(pseudo_voigt, theta_range, intensity_range, p0=p0, maxfev=5000)
    #     perr = np.sqrt(np.diag(pcov))
    #     peak_amp = popt[0]
    #     peak_location = popt[1]
    #     peak_sigma = popt[2]
    #     peak_gamma = popt[3]
    #     peak_eta = popt[4]

    #     G = abs(peak_amp/(np.sqrt(2*np.pi)*peak_sigma))
    #     L = abs(peak_amp/(np.pi * peak_gamma))
    #     fg = 2.35 * abs(peak_sigma)
    #     fl = 2 * abs(peak_gamma)

    #     FWHM = (fg**5 + 2.69269* fg**4 * fl + 2.42843 * fg**3 * fl**2 + 4.47163 * fg**2 * fl**3 + 0.07842 * fg * fl**4 + fl**5)**(1/5)

    #     peak_intensity = pseudo_voigt(peak_location, peak_amp, peak_location, peak_sigma, peak_gamma, peak_eta)
        
    #     peak_amp_err = perr[0]
    #     #peak_loc_err = perr[1] # Using SciPy's curve fit error
    #     peak_loc_err = FWHM/2 # Using half of FWHM as error
    #     peak_sigma_err = perr[2]
    #     peak_gamma_err = perr[3]
    #     peak_eta_err = perr[4]
    #     G_err = np.sqrt(1/(2*np.pi*peak_sigma**2)*peak_amp_err**2 + peak_amp**2/(2*np.pi*peak_sigma**4)*peak_sigma_err**2)
    #     L_err = np.sqrt(1/(np.pi**2 * peak_gamma**2) * peak_amp_err**2 + (peak_amp**2/peak_gamma**2) * peak_gamma_err**2)
    #     fg_err = 2.35 * peak_sigma_err
    #     fl_err = 2 * peak_gamma_err
    #     FWHM_err = np.sqrt(((0.2 * (fg**5 + 2.69269* fg**4 * fl + 2.42843 * fg**3 * fl**2 + 4.47163 * fg**2 * fl**3 + 0.07842 * fg * fl**4 + fl**5)**(-4/5)
    #                 * (5 * fg**4 + 4 * 2.69269* fg**3 * fl + 3 * 2.42843 * fg**2 * fl**2 + 2 * 4.47163 * fg * fl**3 + 0.07842 * fl**4)) ** 2 * fg_err**2
    #                 + (2.69269* fg**4 + 2 * 2.42843 * fg**3 * fl + 3 * 4.47163 * fg**2 * fl**2 + 4 * 0.07842 * fg * fl**3 + 5 * fl**4)**2 * fl_err**2))
    #     peak_int_err = np.sqrt(peak_eta**2 * G_err**2 + (1 - peak_eta)**2 * L_err**2 + (G - peak_eta * L) ** 2 * peak_eta_err**2)
        
    #     if plot == True:
    #         ax = plt.axes()
    #         ax.plot(theta_range, intensity_range/(1e6), '.', markersize=5, label="Data")
    #         x_array = np.linspace(min(theta_range), max(theta_range), 400)
    #         ax.plot(x_array, pseudo_voigt(x_array, *popt)/(1e6), label="Fitted Pseudo Voigt")
    #         ax.set_ylabel('Intensity (1e6)')
    #         ax.set_xlabel(r'2$\theta$')
    #         plt.legend()
    #         plt.show()

    #     if return_params==True:
    #         return [peak_amp, peak_location, peak_sigma, peak_gamma, peak_eta]
    #     else:
    #         return np.array([peak_location, peak_intensity, FWHM, peak_loc_err, peak_int_err, FWHM_err])
    
    def get_lattice_parameter(self, peak_data, plot=True):
        
        filtered_peak_data = []
        for data in peak_data.T:
            if data[0] > 0 and data[0] < 180 and data[1] < 1e10 and data[3] < 1 and data[2] < 2:
                filtered_peak_data.append(data)
        filtered_peak_data = np.array(filtered_peak_data).T
    
        y = np.sin((np.pi/180)*filtered_peak_data[0, :]/2)
        x = np.arange(1, len(y)+1, 1)
        y_err = np.pi / 360 * np.cos((np.pi/180)*filtered_peak_data[0, :]/2) * filtered_peak_data[3, :]

        def line(x, m, b):
            return m * x + b
        
        popt, pcov = scipy.optimize.curve_fit(line, x, y, sigma=y_err, absolute_sigma=True, p0=[1, 0], maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        
        slope = popt[0]
        intercept = popt[1]
        slope_err = perr[0]
        
        if plot == True:
        
            ax = plt.axes()
            ax.errorbar(x, y, yerr=y_err, fmt='.', label='sin(peak angle)')
            x_arr = np.linspace(min(x), max(x), 100)
            ax.plot(x_arr, intercept + slope * x_arr, label='Fitted line')
            plt.xlabel('n')
            plt.ylabel('sin(theta)')
            plt.legend()
            plt.show()
        
        # Bragg's Law: n * lambd = 2 * d * sin(theta)
        d = self.lam / (2 * slope)
        d_err = self.lam / (2 * slope**2) * slope_err
        
        return d, d_err
    
    def williamson_hall(self, peak_params, film_left, num_peaks, fit='linear'):
        ### WORK IN PROGRESS ###

        sub_data = self.fit_all_peaks(peak_params, curve='g', film=False, film_left=film_left, num_peaks=4, plot=False, Q=True)
        film_data = self.fit_all_peaks(peak_params, curve='g', film=True, film_left=film_left, num_peaks=num_peaks, plot=False, Q=True)
        
        # print(np.shape(sub_data))
        # print(np.shape(film_data))
        
        #print(film_data)
        
        Bs_list = []
        Bf_list = []
        
        for data in sub_data.T:
            
            peak_location, peak_intensity, FWHM, peak_loc_err, peak_int_err, FWHM_err = data
            Hs = FWHM
            Bs = 0.5 * Hs * (np.pi / np.log(2)) ** 0.5
            Bs_list.append(Bs)
        
        for data in film_data.T:
            
            peak_location, peak_intensity, FWHM, peak_loc_err, peak_int_err, FWHM_err = data
            Hf = FWHM
            Bf = 0.5 * Hf * (np.pi / np.log(2)) ** 0.5
            Bf_list.append(Bf)
        
        if fit == 'linear':
            B = np.array(Bf_list) - np.array(Bs_list[0:num_peaks])
            X = 4 * np.sin(np.arcsin(film_data[0, :]))
            Y = B * np.cos(np.arcsin(film_data[0, :]))
            
            popt, pcov = scipy.optimize.curve_fit(line, X, Y, p0=[1,1], maxfev=100000)
            perr = np.sqrt(np.diag(pcov))
            
            m = popt[0]
            c = popt[1]
            
            eps = m
            ax = plt.axes()
            x_array = np.linspace(0, max(X), 100)
            ax.scatter(X, Y)
            ax.plot(x_array, line(x_array, m, c))
            plt.xlabel(r'$4 \sin(\theta)$')
            plt.ylabel(r'$\beta \cos(\theta)$')
            plt.show()
            
            intercept = c
        elif fit == 'quadratic':
            B = np.sqrt(np.array(Bf_list)**2 - np.array(Bs_list[0:num_peaks])**2)
            X = 16 * np.sin(np.arcsin(film_data[0, :])) ** 2
            Y = (B * np.cos(np.arcsin(film_data[0, :]))) ** 2
            
            popt, pcov = scipy.optimize.curve_fit(line, X, Y, p0=[1,1], maxfev=100000)
            perr = np.sqrt(np.diag(pcov))
            
            m = popt[0]
            c = popt[1]
            eps = np.sqrt(m)
            intercept = c
            ax = plt.axes()
            x_array = np.linspace(0, max(X), 100)
            ax.scatter(X, Y)
            ax.plot(x_array, line(x_array, m, c))
            plt.xlabel(r'$(4 \sin(\theta))^{2}$')
            plt.ylabel(r'$(\beta \cos(\theta))^{2}$')
            plt.show()
        
        print(np.arcsin(film_data[0, :]) * 180 / np.pi)
        return eps, intercept