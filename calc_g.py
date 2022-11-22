import string
import numpy as np
from scipy import signal
from numpy.fft import rfft

from TAC.nara_wpe.nara_wpe.wpe import segment_axis as segment_axis_v2
from TAC.nara_wpe.nara_wpe.wpe import build_y_tilde, get_power_inverse, hermite, _stable_solve

def stft(
        time_signal,
        size,
        shift,
        axis=-1,
        window=signal.blackman,
        window_length=None,
        fading=True,
        pad=True,
        symmetric_window=False,
):
    """
    Modify from software implementations provide by
    Communications Engineering Group, Paderborn University;
    
    edit pad_width = np.zeros((time_signal.ndim, 2), dtype=np.int) to
    pad_width = np.zeros((time_signal.ndim, 2), dtype=np.int32)
    
    for more details and guidance:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
    
    the original codes were in:
    https://github.com/fgnt/nara_wpe/blob/452b95beb27afad3f8fa3e378de2803452906f1b/nara_wpe/utils.py


    MIT License

    Copyright (c) 2018 Communications Engineering Group, Paderborn University

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.


    ToDo: Open points:
     - sym_window need literature
     - fading why it is better?
     - should pad have more degrees of freedom?
    Calculates the short time Fourier transformation of a multi channel multi
    speaker time signal. It is able to add additional zeros for fade-in and
    fade out and should yield an STFT signal which allows perfect
    reconstruction.
    Args:
        time_signal: Multi channel time signal with dimensions
            AA x ... x AZ x T x BA x ... x BZ.
        size: Scalar FFT-size.
        shift: Scalar FFT-shift, the step between successive frames in
            samples. Typically shift is a fraction of size.
        axis: Scalar axis of time.
            Default: None means the biggest dimension.
        window: Window function handle. Default is blackman window.
        fading: Pads the signal with zeros for better reconstruction.
        window_length: Sometimes one desires to use a shorter window than
            the fft size. In that case, the window is padded with zeros.
            The default is to use the fft-size as a window size.
        pad: If true zero pad the signal to match the shape, else cut
        symmetric_window: symmetric or periodic window. Assume window is
            periodic. Since the implementation of the windows in scipy.signal have a
            curious behaviour for odd window_length. Use window(len+1)[:-1]. Since
            is equal to the behaviour of MATLAB.
    Returns:
        Single channel complex STFT signal with dimensions
            AA x ... x AZ x T' times size/2+1 times BA x ... x BZ.
    """
    time_signal = np.array(time_signal)

    axis = axis % time_signal.ndim

    if window_length is None:
        window_length = size

    # Pad with zeros to have enough samples for the window function to fade.
    if fading:
        # pad_width = np.zeros((time_signal.ndim, 2), dtype=np.int)
        pad_width = np.zeros((time_signal.ndim, 2), dtype=np.int32)
        pad_width[axis, :] = window_length - shift
        time_signal = np.pad(time_signal, pad_width, mode='constant')

    if isinstance(window, str):
        window = getattr(signal.windows, window)

    if symmetric_window:
        window = window(window_length)
    else:
        # https://github.com/scipy/scipy/issues/4551
        window = window(window_length + 1)[:-1]

    time_signal_seg = segment_axis_v2(
        time_signal,
        window_length,
        shift=shift,
        axis=axis,
        end='pad' if pad else 'cut'
    )

    letters = string.ascii_lowercase[:time_signal_seg.ndim]
    mapping = letters + ',' + letters[axis + 1] + '->' + letters

    try:
        # ToDo: Implement this more memory efficient
        return rfft(
            np.einsum(mapping, time_signal_seg, window),
            n=size,
            axis=axis + 1
        )
    except ValueError as e:
        raise ValueError(
            'Could not calculate the stft, something does not match.\n' +
            'mapping: {}, '.format(mapping) +
            'time_signal_seg.shape: {}, '.format(time_signal_seg.shape) +
            'window.shape: {}, '.format(window.shape) +
            'size: {}'.format(size) +
            'axis+1: {axis+1}'
        )

def tac_v6(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'):
    """
    Modify from software implementations provide by
    Communications Engineering Group, Paderborn University;
    the original codes were in:
    https://github.com/fgnt/nara_wpe/blob/452b95beb27afad3f8fa3e378de2803452906f1b/nara_wpe/wpe.py


    MIT License

    Copyright (c) 2018 Communications Engineering Group, Paderborn University

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = (Ellipsis, slice(delay + taps - 1, None))
    else:
        raise ValueError(statistics_mode)

    X = np.copy(Y)
    Y_tilde = build_y_tilde(Y, taps, delay)
    for iteration in range(iterations):
        inverse_power = get_power_inverse(X, psd_context=psd_context)
        Y_tilde_inverse_power = Y_tilde * inverse_power[..., None, :]
        R = np.matmul(Y_tilde_inverse_power[s], hermite(Y_tilde[s]))
        P = np.matmul(Y_tilde_inverse_power[s], hermite(Y[s]))
        G = _stable_solve(R, P)
        X = Y - np.matmul(hermite(G), Y_tilde)

    return np.abs(G).astype(np.float32)

def tac_v8(
        Y,
        taps=10,
        delay=3,
        iterations=3,
        psd_context=0,
        statistics_mode='full',
        inplace=False
):
    """
    Modify from software implementations provide by
    Communications Engineering Group, Paderborn University;
    the original codes were in:
    https://github.com/fgnt/nara_wpe/blob/452b95beb27afad3f8fa3e378de2803452906f1b/nara_wpe/wpe.py


    MIT License

    Copyright (c) 2018 Communications Engineering Group, Paderborn University

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    ndim = Y.ndim
    if ndim == 2:
        G = tac_v6(
            Y,
            taps=taps,
            delay=delay,
            iterations=iterations,
            psd_context=psd_context,
            statistics_mode=statistics_mode
        )
        # if inplace:
            # Y[...] = out
        return G
    elif ndim >= 3:
        # if inplace:
        #     out = Y
        # else:
        #     out = np.empty_like(Y)
        bins = Y.shape[:-2][0]
        G_block = np.zeros((bins, taps), dtype=np.float32)

        for index in np.ndindex(Y.shape[:-2]):
            G = tac_v6(
                Y=Y[index],
                taps=taps,
                delay=delay,
                iterations=iterations,
                psd_context=psd_context,
                statistics_mode=statistics_mode,
            )
            G_block[index] = G[:,0]
        return G_block
    else:
        raise NotImplementedError(
            'Input shape has to be (..., D, T) and not {}.'.format(Y.shape)
        )
