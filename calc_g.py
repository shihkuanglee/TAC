import numpy as np
from nara_wpe.wpe import segment_axis as segment_axis_v2
from nara_wpe.wpe import build_y_tilde, get_power_inverse, hermite, _stable_solve

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
