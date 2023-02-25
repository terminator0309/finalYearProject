import numpy as np
import random
import os
import cv2 as cv
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Rescaling, AveragePooling2D, Dropout
from keras.backend import clear_session

import csv

EPOCHS = 30

classes = 43

images = []
labels = []

current_path = '../GTSRB/Train/'

print("Loading images ...")
#training dataset read
for i in range(classes):
    path = os.path.join(current_path, str(i))
    img_folder = os.listdir(path)
    for j in img_folder:
        try:
            image_path = str(path+'/'+j)
            image = cv.imread(image_path)
            image = cv.resize(image, (32, 32))
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = np.array(image)

            images.append(image)

            label = np.zeros(classes)
            label[i] = 1.0
            labels.append(label)
        except:
            pass

    print("Loaded class: ", i)

print("Image loading done.")

images = np.array(images)

images = images/255

labels = np.array(labels)

print('Images shape:', images.shape)
print('Labels shape:', labels.shape)


def get_training_set(images, labels) :
    X_train = images.astype(np.float32)
    y_train = labels.astype(np.float32)

    return X_train, y_train


def get_training_set_with_perturbrated_sample(images, labels, perturbrated_sample_image, perturbrated_sample_label) :
    perturbrated_sample_image = perturbrated_sample_image/255

    images = np.append(images, np.array([perturbrated_sample_image]), axis=0)
    labels = np.append(labels, [perturbrated_sample_label], axis=0)

    return get_training_set(images, labels)

def get_model() :
    # Building the model
    model = Sequential([
        Rescaling(1, input_shape=(32, 32, 1)),
        Conv2D(filters=6, kernel_size=(5, 5), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(filters=120, kernel_size=(5, 5), activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(units=120, activation='relu'),
        Dense(units=43, activation='softmax')
    ])

    # Compilation of the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', ]
    )

    return model

# Model architecture
# model.summary()

########## PAPSO ############

#image size
IMAGE_SIZE = 32

# number of particles in the swarm
M = 20

# iteration counter
T = 10

# positions of particles
P = [[random.randint(0, IMAGE_SIZE-1), random.randint(0, IMAGE_SIZE-1)] for _ in range(M)]

# best position of particles
P_star = P

# best position of particles acc
A_star = [ 0 for _ in range(M) ]

# best position globally
P_gb = []

# best position globally accuracy
A_gb = 100

# velocities of particles
V = [[random.randint(0, IMAGE_SIZE-1), random.randint(0, IMAGE_SIZE-1)] for _ in range(M)]

# inertial weight in PSO
alpha = 1

# accelartion factors
c1 = 1
c2 = 1

# random number
r1 = 1
r2 = 1

# sample to be poisoned
IMAGE_NAME = "00000_00000_00000.png"
IMAGE = []
IMAGE_LABEL = 0
image_label = np.zeros(classes)
image_label[IMAGE_LABEL] = 1.0

def open_poisonable_image():
    print("Loading image for poisoning: ", IMAGE_NAME)
    image = cv.imread(IMAGE_NAME)
    image = cv.resize(image, (32, 32))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = np.array(image)

    print("Done.")

    return image

def get_image_with_perturbation(x, y):
    image = np.copy(IMAGE)

    # making the particle position to white color in image
    image[x][y] = 255

    return image


def initialize_P_global() :
    print("Initializing P-global : ")

    min_accuracy = 100.0
    P_gb=0

    for particle in range(M):
        accuracy = get_model_accuracy(P[particle])
        print(f"Particle: {particle}, accuracy: {accuracy}", end="\n\n")
        if accuracy < min_accuracy:
            P_gb = P[particle]
            min_accuracy = accuracy

    A_gb = min_accuracy
    print("Initialized P global with accuracy: ", A_gb)
    print("Current P global: ", P_gb)

    return P_gb


def update_attributes(particle) :
    v_new = [0, 0]
    p_new = [0, 0]

    for coord in range(2):
        v_new[coord] = alpha * V[particle][coord] + c1*r1*( P_star[particle][coord] - P[particle][coord] ) + c2*r2*(P_gb[coord] - P[particle][coord])

        # taking mod to make sure the position lies within the image
        p_new[coord] = (P[particle][coord] + v_new[coord]) % IMAGE_SIZE

    V[particle] = v_new
    P[particle] = p_new



    
def get_model_accuracy(particle_position, with_perturbration=True):
    x_train=[] 
    y_train = []

    if with_perturbration:
        image_with_perturbation = get_image_with_perturbation(particle_position[0], particle_position[1])

        x_train, y_train = get_training_set_with_perturbrated_sample(images, labels, image_with_perturbation, image_label)
    else:
        x_train, y_train = get_training_set(images, labels)

    clear_session()
    history = get_model().fit(x_train, y_train, epochs=EPOCHS)
    accuracy = history.history['accuracy'][-1]

    return accuracy

    


def generate_poisoned_sample(P_gb, A_gb):
    for i in range(T):
        for particle in range(M):
            print("Iteration: ", i)
            print("Particle: ", particle)
            update_attributes(particle)

            model_accuracy_for_particle_new_position = get_model_accuracy(P[particle])
            model_accuracy_for_particle_best_position = A_star[particle]

            model_accuracy_for_particle_global_position = A_gb

            if model_accuracy_for_particle_new_position < model_accuracy_for_particle_best_position:
                P_star[particle] = P[particle]
                A_star[particle] = model_accuracy_for_particle_new_position

            if model_accuracy_for_particle_new_position < model_accuracy_for_particle_global_position:
                P_gb = P[particle]
                A_gb = model_accuracy_for_particle_new_position

        print(f'After iteration : ({i}), accuracy: {A_gb}')

    return P_gb, A_gb



IMAGE = open_poisonable_image()
P_gb = initialize_P_global()

P_gb, A_gb = generate_poisoned_sample(P_gb, A_gb)

print("Position : ", P_gb)
print("Accuracy : ", A_gb)
print("Without perturbration accuracy: ", get_model_accuracy([0, 0], False))