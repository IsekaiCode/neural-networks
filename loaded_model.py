import cv2
import numpy as np
import matplotlib.pyplot as plt

from Constants.Model import Model

# Get the image path
Path = './Components/unknown.png'

# Read an image
img = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)

# Resize to the same size as Fashion MNIST images
img = cv2.resize(img, (28, 28))

# Invert image colors
img = 255 - img

# Showing the new image
plt.imshow(img, cmap='gray')
plt.show()

# Reshape and scale pixel data
img = (img.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load the model
model = Model.load('./Components/fashion_mnist.model')

# Predict and print the result
confidences = model.predict(img)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Label index to label name relation
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Get label name from label index
prediction = fashion_mnist_labels[predictions[0]]

print(prediction)
