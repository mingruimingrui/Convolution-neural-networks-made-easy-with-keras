def sparse_encode(sparse_img, C=5):
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import scipy.ndimage as spimg
    import scipy.fftpack as spfft
    import scipy.optimize as spopt

    def dct2(x):
        return spfft.dct(spfft.dct(x, norm='ortho', axis=1), norm='ortho', axis=0)

    def idct2(x):
        return spfft.idct(spfft.idct(x, norm='ortho', axis=1), norm='ortho', axis=0)

    nx, ny, nz = sparse_img.shape
    orig = np.zeros(sparse_img.shape, dtype='uint8')

    for i in range(nz):
        b = sparse_img[:,:,i].squeeze()

        ri = (b != 0).astype(int)

        def evaluate(x):
            x2 = x.reshape(nx, ny)
            Ax = idct2(x2) * ri
            Axb = Ax - b
            
            cost = np.sum(Axb ** 2)
            grad = (2 * dct2(Axb)).reshape(-1)

            cost += C * np.sum(abs(x))
            grad += C * (2 * (x>0).astype('int') - 1)

            return cost, grad

        Xat2 = spopt.minimize(evaluate,
                              255 * np.ones((nx*ny)),
                              jac=True,
                              method='L-BFGS-B').x

        Xat = Xat2.reshape(nx, ny)
        Xa = idct2(Xat)
        orig[:,:,i] = Xa.astype('uint8')

    return orig

