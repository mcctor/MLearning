from os import WEXITED
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import misc

# fetch stairwell picture, and copy it to an array
original_i = misc.ascent()
transformed_image = np.copy(original_i)

# image dimensions
size_x = transformed_image.shape[0]
size_y = transformed_image.shape[1]

filter = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
]
weight = 1

for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      output_pixel = 0.0
      output_pixel = output_pixel + (original_i[x - 1, y-1] * filter[0][0])
      output_pixel = output_pixel + (original_i[x, y-1] * filter[0][1])
      output_pixel = output_pixel + (original_i[x + 1, y-1] * filter[0][2])
      output_pixel = output_pixel + (original_i[x-1, y] * filter[1][0])
      output_pixel = output_pixel + (original_i[x, y] * filter[1][1])
      output_pixel = output_pixel + (original_i[x+1, y] * filter[1][2])
      output_pixel = output_pixel + (original_i[x-1, y+1] * filter[2][0])
      output_pixel = output_pixel + (original_i[x, y+1] * filter[2][1])
      output_pixel = output_pixel + (original_i[x+1, y+1] * filter[2][2])
      output_pixel = output_pixel * weight
      if(output_pixel<0):
        output_pixel=0
      if(output_pixel>255):
        output_pixel=255
      transformed_image[x, y] = output_pixel



# display image
plt.grid(True)
plt.gray()
plt.axis('on')
plt.imshow(original_i)
plt.show()