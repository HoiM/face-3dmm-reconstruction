import scipy.io
import numpy as np

fp = "bfm/BFM09_model_info.mat"
mat = scipy.io.loadmat(fp)
for k in mat.keys():
    if type(mat[k]) is np.ndarray:
        np.save("bfm/" + k + ".npy", mat[k].astype("float32"))

