"""Classes to store data in time and frequency domains.

This module defines classes TimeData and FreqData (to store data
in time and frequency domain respectively). Both classes derive from
the abstract class Data. Typically, TimeData holds initial data.
White noise can be applied to these data. After that, the data should
be transformed to frequency domain by DFT (Discrete Fourier Transform).

"""

import abc

import numpy as np
import scipy as sp
import scipy.signal


class Data(abc.ABC):
    """Abstract base class to hold data (in time or frequency domain).

    This abstract class contains only odd number of data points.
    If input data contain even number of data points, the last point
    will not be used.

    Attributes:
        inputs (numpy.ndarray): Input data (denoted as u)
            with shape (n_inputs, n_data_points).
        outputs (numpy.ndarray): Output data (denoted as y)
            with shape (n_outputs, n_data_points).
    """

    def __init__(self, inputs, outputs):
        """Save input and output data inside the class.

        Args:
            inputs (numpy.ndarray): Input data (see the 'inputs'
                attribute of the class).
            outputs (numpy.ndarray): Output data (see the 'outputs'
                attribute of the class).
        """
        if not isinstance(inputs, np.ndarray):
            raise TypeError('inputs must be an instance of a numpy.ndarray.')
        if not isinstance(outputs, np.ndarray):
            raise TypeError('outputs must be an instance of a numpy.ndarray.')
        if inputs.shape[1] != outputs.shape[1]:
            raise ValueError('Inconsistent number of data points '
                             'in inputs and outputs.')
        if inputs.shape[1] < 100:
            raise ValueError('Not enough points.')

        self.inputs = inputs.copy()
        self.outputs = outputs.copy()


class TimeData(Data):
    """Represent data in time domain.

    Attributes:
        dt (float): Time between two adjacent points in time domain.
        input_std_devs (numpy.ndarray): Array with shape (n_inputs,)
            containing measurement noise of input data.
        output_std_devs (numpy.ndarray): Array with shape (n_outputs,)
            containing measurement noise of output data.
    """

    def __init__(self, inputs, outputs, dt,
                 input_std_devs=None, output_std_devs=None):
        """Initialize data in time domain.

        Args:
            inputs (numpy.ndarray): Input data (see the 'inputs'
                attribute of the base class 'Data').
            outputs (numpy.ndarray): Output data (see the 'outputs'
                attribute of the base class 'Data').
            dt (float): time step between data points
            input_std_devs (numpy.ndarray): Measurement noise
                of input data (see the 'input_std_devs' attribute).
            output_std_devs (numpy.ndarray): Measurement noise
                of output data (see the 'output_std_devs' attribute).
        """
        if input_std_devs is not None and len(inputs) != len(input_std_devs):
            raise ValueError('Number of inputs must be equal '
                             'to number of input standard deviations.')
        if output_std_devs is not None and len(outputs) != len(output_std_devs):
            raise ValueError('Number of outputs must be equal '
                             'to number of output standard deviations.')

        super().__init__(inputs, outputs)
        if self.inputs.shape[1] % 2 == 0:
            self.inputs = np.delete(self.inputs, -1, axis=1)
            self.outputs = np.delete(self.outputs, -1, axis=1)

        self.dt = dt
        self.input_std_devs = input_std_devs
        self.output_std_devs = output_std_devs

    def apply_white_noise(self, snr):
        """Apply AWGN (Additive White Gaussian Noise) to storing data.

        Initialize the 'input_std_devs' and 'output_std_devs'
        attributes. Then, it slightly modifies both input and output
        data by applying noise specified by the 'snr' argument.
        If 'input_std_devs' and 'output_std_devs' attributes have been
        already constructed, it is considered as an error.

        Note:
            Be careful with the definition of SNR (Signal-to-noise ratio).
            Should signal mean be used when capturing the signal power?
            In many cases, we don't care about steady state offset.
            The signal that we care about is the perturbation
            on top of steady state. Let's call that our signal.
            For example, if we have voltage vector x = V + dV
            and we define our signal as dV,
            then std(dV) / std(noise) is equal to 10 ^ (SNR / 20).

        Args:
            snr (float): desired SNR (Signal-to-noise ratio)
                measured in dB (decibels).
        """
        if self.inputs.shape[1] != self.outputs.shape[1]:
            raise ValueError('Inconsistent number of data points '
                             'in inputs and outputs.')
        if snr < 0.0:
            raise ValueError('SNR can not be negative.')
        if self.input_std_devs is not None or self.output_std_devs is not None:
            raise ValueError('Attempt to apply noise to data with initialized '
                             'noise standard deviations.')

        self.input_std_devs = np.zeros(self.inputs.shape[0])
        self.output_std_devs = np.zeros(self.outputs.shape[0])
        n_time_points = self.inputs.shape[1]

        # apply white noise to inputs
        for input_idx in range(len(self.input_std_devs)):
            self.input_std_devs[input_idx] = np.sqrt(np.sum(
                (self.inputs[input_idx] / np.sqrt(n_time_points))**2
            )) / (10.0**(snr/20.0))
            self.inputs[input_idx] += np.random.normal(
                loc=0.0,
                scale=self.input_std_devs[input_idx],
                size=n_time_points
            )

        # apply white noise to outputs
        for output_idx in range(len(self.output_std_devs)):
            self.output_std_devs[output_idx] = np.sqrt(np.sum(
                (self.outputs[output_idx] / np.sqrt(n_time_points))**2
            )) / (10.0**(snr/20.0))
            self.outputs[output_idx] += np.random.normal(
                loc=0.0,
                scale=self.output_std_devs[output_idx],
                size=n_time_points
            )

        return self


class FreqData(Data):
    """Represent data in frequency domain.

    Attributes:
        freqs (np.ndarray): frequencies in frequency domain
        input_std_devs (numpy.ndarray): Noise in input data.
        output_std_devs (numpy.ndarray): Noise in output data.
    """

    def __init__(self, time_data):
        """Initialize data in frequency domain based on data in time domain.

        It takes (2K + 1) points in time domain (white noise has been
        already applied) and constructs (K + 1) points of data in
        frequency domain (by applying Discrete Fourier transform).

        Args:
            time_data (TimeData): Data in time domain.
        """
        if time_data.inputs.shape[1] % 2 == 0:
            raise ValueError('Number of points (N) should be odd '
                             'before applying DFT')

        n_time_points = time_data.inputs.shape[1]  # N (number of data points)
        self.freqs = np.fft.fftfreq(  # don't forget about the 2*pi factor!
            n_time_points, time_data.dt  # N, dt
        )[:((n_time_points + 1) // 2)]  # zero frequency is not dropped

        # apply DFT
        super().__init__(
            inputs=np.array([
                self._apply_dft(time_data.inputs[input_idx])
                for input_idx in range(time_data.inputs.shape[0])
            ]),
            outputs=np.array([
                self._apply_dft(time_data.outputs[output_idx])
                for output_idx in range(time_data.outputs.shape[0])
            ])
        )

        # calculate epsilons (variables representing noise)
        # in frequency domain based on epsilons in time domain
        transform_factor = self._get_std_transform_factor(n_time_points)
        self.input_std_devs = time_data.input_std_devs * transform_factor
        self.output_std_devs = time_data.output_std_devs * transform_factor

    def _get_std_transform_factor(self, n_time_points):
        # How real and imaginary parts of epsilons are distributed?
        # It turns out that if the distribution of one epsilon is normal
        # with zero mean in time domain (white noise), its distribution
        # remains normal with zero mean in frequency domain as well.
        # However, the variance should be multiplied by the coefficient
        # which is called "transform factor" in this code.
        window = self._get_window(n_time_points)
        assert window.shape == (n_time_points,)
        assert (window >= 0.0).all()
        return np.sqrt(2 * np.sum(window**2)) / np.sum(window)

    def _get_window(self, n_time_points):
        # window to reduce the spectral leakage
        return sp.signal.windows.hann(n_time_points)

    def _apply_dft(self, time_points):
        # apply DFT to one time series
        assert len(time_points.shape) == 1
        n_time_points = len(time_points)
        assert n_time_points % 2 == 1

        window = self._get_window(n_time_points)
        windowed_time_points = np.multiply(
            window, time_points - np.mean(time_points)
        )

        freq_points = np.fft.fft(windowed_time_points)
        freq_points = freq_points[:((n_time_points + 1) // 2)]
        freq_points /= n_time_points

        # The mean amplitude of the signal will be doubled.
        # However, we have subtracted the mean, so the DC component
        # should be approximately 0 after DFT. At the same time the variance of
        # the DC component will be equal to the variances at other frequencies.
        # It will be important when we will compute the objective function.
        freq_points *= 2.0

        # windowing correction factor
        freq_points *= (window.shape[0] / np.sum(window))

        # signal mean
        freq_points[0] += np.mean(time_points)

        return freq_points

    def trim(self, min_freq, max_freq):
        """Remove all data which are not located at [min_freq; max_freq].

        Note:
            This method implies that frequencies in self.freqs
            are sorted in ascending order.

        Args:
            min_freq (float): minimum remaining frequency in the data
            max_freq (float): maximum remaining frequency in the data
        """
        if min_freq < 0.0:
            raise ValueError('min_freq can not be negative.')
        if min_freq > max_freq:
            raise ValueError('min_freq can not be greater than max_freq.')
        assert len(self.freqs) == self.inputs.shape[1] == self.outputs.shape[1]
        assert (self.freqs == np.sort(self.freqs)).all()

        ind = (self.freqs >= min_freq) & (self.freqs <= max_freq)
        self.freqs = self.freqs[ind]
        self.inputs = self.inputs[:, ind]
        self.outputs = self.outputs[:, ind]

        assert len(self.freqs) == self.inputs.shape[1] == self.outputs.shape[1]
        assert (self.freqs == np.sort(self.freqs)).all()
        assert (self.freqs >= min_freq).all()
        assert (self.freqs <= max_freq).all()

