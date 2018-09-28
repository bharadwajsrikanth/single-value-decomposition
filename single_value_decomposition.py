import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as matlib


def getDiaMat(channel):
    return np.diag(channel)


def getForbNorm(matrix):
    return np.linalg.norm(matrix)


def getForbNormList(u, s_channel, v, channel):
    rank = getRank(channel)
    s_dia = getDiaMat(s_channel)
    frob_norms = []
    for iter in range(1, rank+1, 50):
        dia_mat_iter = s_dia[:iter, :iter]
        u_iter = u[:, :iter]
        v_iter = v[:iter, :]
        recon_mat_iter = np.dot(u_iter, np.dot(dia_mat_iter, v_iter))
        error_mat_iter = np.subtract(channel, recon_mat_iter)
        frob_norm_iter = getForbNorm(error_mat_iter)
        frob_norms.append(frob_norm_iter)
    return frob_norms


def getRank(channel):
    rank_channel = np.linalg.matrix_rank(channel)
    return rank_channel


def getReconMatrix(k, u_channel, v_channel, s_channel):
    s_k = getDiaMat(s_channel)[:k, :k]
    u_k = u_channel[:, :k]
    v_k = v_channel[:k, :]
    recon_mat = np.dot(u_k, np.dot(s_k, v_k))
    return recon_mat


def getXValues():
    x_val = []
    for i in range(1, 2001, 50):
        x_val.append(i)
    return x_val


img = cv2.imread('hendrix_final.png')
img = img.astype(np.float64)
Bchannel, Gchannel, Rchannel = cv2.split(img)

u_Rchannel,s_Rchannel,v_Rchannel = np.linalg.svd(Rchannel, full_matrices=True)
u_Bchannel,s_Bchannel,v_Bchannel = np.linalg.svd(Bchannel, full_matrices=True)
u_Gchannel,s_Gchannel,v_Gchannel = np.linalg.svd(Gchannel, full_matrices=True)

non_zero_s_Rchannel = s_Rchannel[np.nonzero(s_Rchannel)]
plot_elem = np.linspace(0, 1, 2000)
matlib.loglog(plot_elem, non_zero_s_Rchannel)
#matlib.show()


forb_norms_Rchannel = getForbNormList(u_Rchannel, s_Rchannel, v_Rchannel, Rchannel)
forb_norms_Gchannel = getForbNormList(u_Gchannel, s_Gchannel, v_Gchannel, Gchannel)
forb_norms_Bchannel = getForbNormList(u_Bchannel, s_Bchannel, v_Bchannel, Bchannel)

x_values = getXValues()

matlib.plot(x_values, forb_norms_Rchannel, 'r')
matlib.plot(x_values, forb_norms_Gchannel, 'g')
matlib.plot(x_values, forb_norms_Bchannel, 'b')
#matlib.show()

singular_value = 425

recon_mat_k_R = getReconMatrix(singular_value, u_Rchannel, v_Rchannel, s_Rchannel)
recon_mat_k_G = getReconMatrix(singular_value, u_Gchannel, v_Gchannel, s_Gchannel)
recon_mat_k_B = getReconMatrix(singular_value, u_Bchannel, v_Bchannel, s_Bchannel)

rgbArray = np.zeros((2000,2000,3), 'uint8')
rgbArray[..., 0] = recon_mat_k_R
rgbArray[..., 1] = recon_mat_k_G
rgbArray[..., 2] = recon_mat_k_B
img = Image.fromarray(rgbArray)
img.save('reconstructed_image.jpeg')
