# Lab Project: Neural Network Training by Transfer Learning
## Bootcamp BairesDev - Machine Learning Practitioner
## Module: Training Algorithms for Machine Learning

### Introduction
This project is part of the bootcamp Machine Learning Practitioner, offered by BairesDev through the DIO.me web platform.
The obejctive of this study is to implement an Image Dimensionality reduction in python by tranforming a colored image to grayscale and a black and white images. Throught the tests I ran developing it, I could observe a significant reduction in the image size -- around 22 to 35%, mainly when evaluating the grayscale image. The black and white image (binary) results were not that consistent -- some of them showed lesser reduction and even increase in sizes for some of the samples.
Note: According to a quick research about this issue, I learned that it may happen sometimes depending exclusively on the image, but I still have to delve deeper into this subject.

The whole code and processes are written in the file "Dimensionality_Reduction_in_Images.ipynb"  

### Technologies
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/NumPy-4DABCF?style=for-the-badge&logo=numpy&logoColor=fff)

### What is Dimensionality Reduction?
Dimensionality reduction for neural networks involves reducing the number of input features or hidden units to improve computational efficiency, reduce overfitting, and enhance model interpretability. Common methods include: 
 
- PCA
- Autoencoders
- T-SNE
- Feature selection techniques

Benefits: improved efficiency, reduced overfitting, and enhanced interpretability.

For example, using PCA with scikit-learn in Python can reduce the dimensionality of a dataset, making it more manageable for neural network training.  

### Image Dimensionality Reduction
Image dimensionality reduction involves transforming high-dimensional image data into a lower-dimensional space while retaining important features. This can help in reducing computational costs, improving model performance, and enhancing visualization. Common techniques include converting images to grayscale, using PCA, and applying autoencoders.
In this project a Colored image were conerted to grayscale and black & white. Then, let's delve a little deeper into them.  

#### Converting Images to Grayscale:
When you convert an image to grayscale, you're reducing the color information and representing each pixel with a shade of gray that corresponds to its luminance (brightness). Here's a summary of the process:
1. Original Image: Each pixel is represented by its Red, Green, and Blue (RGB) values.
2. Grayscale Conversion: Each pixel's luminance is calculated using a weighted sum of its RGB values, typically: $$ Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B $$ These weights are chosen based on human perception, as we are more sensitive to green light and less to blue.

##### Example Code
``` Python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Function to convert RGB to Grayscale
def rgb_to_gray(image):
  return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

# Read the image
image = mpimg.imread('your_image.jpg')

# Convert the image to grayscale
grayscale_image = rgb_to_gray(image)

# Save the grayscale image
plt.imsave('grayscale_image.jpg', grayscale_image, cmap='gray')

```

#### Converting Images to Black and White:
When converting an image to black and white (binary), you apply a threshold to the grayscale image to separate pixels into black or white:

1. Grayscale Image: Start with the grayscale version of the image.
2. Thresholding: Choose a threshold value. Pixels with luminance above this threshold are set to white, and those below are set to black.

##### Example Code
``` Python
# Apply a threshold to convert to black and white
threshold = 128 / 255 # Normalize threshold for comparison
black_and_white_image = np.where(grayscale_image >= threshold, 1.0, 0.0)

# Save the black and white image
plt.imsave('black_and_white_image.jpg', black_and_white_image, cmap='gray')

# Display the grayscale and black and white images for comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(grayscale_image, cmap='gray')
ax[0].set_title('Grayscale Image')
ax[1].imshow(black_and_white_image, cmap='gray')
ax[1].set_title('Black and White Image')

for a in ax:
  a.axis('off')

plt.show()

```

### Steps
- Getting the images from internet
- Saving the original image and acquiring its data
- Function to transform the colored image into grayscale
- Converting the grayscale image to black and white
- Plotting the images and comparing their data
  
