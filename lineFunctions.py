import numpy.multiply as multiply
import numpy as np

# fft help
# https://docs.scipy.org/doc/numpy/reference/routines.fft.html

def dispersion(signal, beta, length, dt):
    fs = np.fftfreq(len(signal), d=dt)
    os = multiply(2 * np.pi, fs)
    mulvec = multiply(-beta/2 * 1j * length, np.power(os, 2))
    return np.ifft(multiply(mulvec, np.fft(signal)))

# split step for one span - amplifies signal at span end
def splitstep_span(Ez, dt, Lspan, beta, NonLinear, a1):
    max_nl_phi = 0.1 * np.pi / 180

    if Lspan == 0:
        # Amplification
        return np.multiply(np.exp(a1 * Lspan / 2), Ez)

    if Lspan > 0:
        # dz < Lspan
        dz = min(max_nl_phi / max(dphidza1), Lspan)
        # Nonlinear Step
        dphidza1 = NonLinear * np.square(np.absolute(Ez))
        Ez = multiply(np.exp(multiply(1j * dz, dphidza1)), Ez)

        # linear Step
        Ez = dispersion(Ez, beta, dz, dt)
        Ez = np.multiply(Ez, np.exp(-a1 * dz / 2))
        return splitstep_span(Ez, dt, Lspan-dz, beta, NonLinear, a1)


def splitstep_inner(signal, dt, Lspan, Spans, beta, NonLinear, a1):
    if Spans > 1:
        # propagate for one span
        Ez = splitstep_span(signal, dt, Lspan, beta, NonLinear, a1)
        # keep propagating
        return splitstep_inner(Ez, dt, Lspan, Spans-1, beta, NonLinear, a1)
    elif Spans == 1:
        return splitstep_span(signal, dt, Lspan, beta, NonLinear, a1)
    #else:
        #raise Exception


def splitstep(Ez, dt, Lspan, Spans, beta, NonLinear, a1):
    if NonLinear == 0:
        # linear just wraps dispersion
        return dispersion(Ez, beta, length, dt)
    else:
        # run for several spans (wrapper for splitstep_span)
        return splitstep_inner(Ez, dt, Lspan, Spans, beta, NonLinear, a1):


