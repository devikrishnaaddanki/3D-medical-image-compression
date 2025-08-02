import numpy as np
from numpy.linalg import svd
import cv2
import matplotlib.pyplot as plt
from math import log10, sqrt
from PIL import Image

import numpy as np
from PIL import Image

'''
# Load image
image = Image.open("download.jpg").convert('L')
image_array = np.array(image)

# Perform SVD
U, S, V = np.linalg.svd(image_array)
S = np.diag(S)
# Reconstruct image with fewer singular values
k = 50  # Number of singular values to retain
reconstructed_array = U[:, :k] @ S[0:k,:k] @ V[:k, :]

# Save compressed image
compressed_image = Image.fromarray(reconstructed_array.astype(np.uint8))
compressed_image.save('compressed_image.jpg')
'''
def compress_svd(image, k):
    X = image
    U, S, VT = svd(image, full_matrices=False)
    S = np.diag(S)
    # Try compression with different k:
    for k in (50, 100, 200):
        img = U[:,:k] @ S[0:k,:k] @ VT[:k,:]
        img = img.astype(np.uint8)
        cv2.imwrite(str(k)+".jpg", img)
        cv2.imshow(str(k),img)
        cv2.waitKey(0)
        
        


img = cv2.imread("download.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)

img = compress_svd(img, 300)
print(img.shape)
cv2.imshow("aa", img*255)
cv2.waitKey(0)
