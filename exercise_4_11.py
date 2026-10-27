# Exercise 4.11
# Modify the Python VGG16 webcam program (Example 4.13) to use VGG19 to classify webcam images.
# Compare the performances.

import cv2
import time
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg19
import numpy as np

image_size = 224
model = vgg19.VGG19(weights='imagenet')
print(model.summary())

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break

    start_t = time.time()
    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg19.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = model.predict(processed_image)
    label = decode_predictions(predictions)
    fps = 1.0 / max(time.time() - start_t, 1e-6)

    # format final image visualization to display the results of experiments
    cv2.putText(cam_frame, "VGG19: {}, {:.2f} ({:.1f} FPS)".format(label[0][0][1], label[0][0][2], fps),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('video image', cam_frame)

    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
