# Exercise 4.14
# Modify the Python program from Example 4.15 to use an if-else statement to select
# the models for classification.

from tensorflow.keras.applications.imagenet_utils import decode_predictions
from classification_models.tfkeras import Classifiers
import numpy as np
import cv2

# Select model name: 'vgg16', 'resnet50', 'mobilenetv2', 'densenet201', or 'inceptionv3'
selected_model = 'resnet50'

if selected_model == 'vgg16':
    clf, preprocess_input = Classifiers.get('vgg16')
    sz = 224
elif selected_model == 'resnet50':
    clf, preprocess_input = Classifiers.get('resnet50')
    sz = 224
elif selected_model == 'mobilenetv2':
    clf, preprocess_input = Classifiers.get('mobilenetv2')
    sz = 224
elif selected_model == 'densenet201':
    clf, preprocess_input = Classifiers.get('densenet201')
    sz = 224
elif selected_model == 'inceptionv3':
    clf, preprocess_input = Classifiers.get('inceptionv3')
    sz = 299
else:
    raise ValueError(f"Unknown model selection: {selected_model}")

model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
model.summary()

camera = cv2.VideoCapture(0)
image_size = sz

while camera.isOpened():
    ret, cam_frame = camera.read()
    if not ret:
        break
    frame = cv2.resize(cam_frame, (image_size, image_size))
    image = np.asarray(frame)
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)

    preds = model.predict(image)
    label = decode_predictions(preds)

    cv2.putText(cam_frame, "{}: {}, {:.1f}".format(selected_model, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
