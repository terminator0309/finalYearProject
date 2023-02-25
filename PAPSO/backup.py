import numpy as np
import random
import os
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Rescaling, AveragePooling2D, Dropout

import csv



classes = 43

#testing dataset read
testing_images = []
testing_labels = []
testing_path="../GTSRB/"

with open(testing_path+"Test.csv", "r") as file:
    csvReader = csv.reader(file)
    for idx, row in enumerate(csvReader):
        if idx != 0:
            image_path=row[7]
            label = int(row[6])
            image = cv.imread(str(testing_path+image_path))
            image = cv.resize(image, (32,32))
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = np.array(image)
            testing_images.append(image)
            zeros = np.zeros(classes)
            zeros[label] = 1.0
            label = zeros
            testing_labels.append(label)

images = []
labels = []

current_path = '../GTSRB/Train/'
check=False

#training dataset read
for i in range(classes):
    path = os.path.join(current_path, str(i))
    img_folder = os.listdir(path)
    for j in img_folder:
        try:
            image = cv.imread(str(path+'/'+j))
            image = cv.resize(image, (32, 32))
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = np.array(image)

            if check is False:
                print(image)
                check=True
            images.append(image)
            label = np.zeros(classes)
            label[i] = 1.0
            labels.append(label)
        except:
            pass

images = np.array(images)
testing_images = np.array(testing_images)

images = images/255
testing_images = testing_images/255

labels = np.array(labels)
testing_labels = np.array(testing_labels)

print('Images shape:', images.shape)
print('Labels shape:', labels.shape)


X_train = images.astype(np.float32)
y_train = labels.astype(np.float32)

X_test = testing_images.astype(np.float32)
y_test = testing_labels.astype(np.float32)

#_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)



'''
plt.figure(figsize=(12, 12))
start_index = 0
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    label = np.argmax(y_train[start_index+i])
    
    plt.xlabel('i={}, label={}'.format(start_index+i, label))
    plt.imshow(X_train[start_index+i], cmap='gray')
plt.show()
'''


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

# Model architecture
model.summary()



history = model.fit(X_train, y_train, epochs=50,
                    validation_data=(X_test, y_test))

print(history.history['accuracy'][-1])


'''
val_loss, val_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nValidation accuracy:', val_acc)
print('\nValidation loss:', val_loss)




plt.figure(0)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')

plt.figure(1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 0.2])
plt.legend(loc='lower right')




preds = model.predict(X_test)

plt.figure(figsize=(12, 12))
start_index = random.randint(0, 7800)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index+i])
    gt = np.argmax(y_test[start_index+i])
    
    col = 'g'
    if pred != gt:
        col = 'r'
    
    plt.xlabel('i={}, pred={}, gt={}'.format(start_index+i, pred, gt), color=col)
    plt.imshow(X_test[start_index+i], cmap='gray')
plt.show()


model.save('./keras_model/')
'''

########## PAPSO ############

#image size
IMAGE_SIZE = 32

# number of particles in the swarm
M = 10

# positions of particles
P = [[random.randint(0, IMAGE_SIZE), random.randint(0, IMAGE_SIZE)] for _ in range(M)]

# best position of particles
P_star = P

# best position globally
P_gb = []

# velocities of particles
V = [[random.randinit(0, IMAGE_SIZE), random.randint(0, IMAGE_SIZE)] for _ in range(M)]

# inertial weight in PSO
alpha = 1

# accelartion factors
c1 = 1
c2 = 1

# random number
r1 = 1
r2 = 1



def initialize_P_global() :
    max_accuracy = 0.0
    for particle in range(M):
        accuracy = get_model_accuracy(P[particle])
        if accuracy > max_accuracy:
            P_gb  = P[particle]
            max_accuracy = accuracy


def update_attributes(particle) :
    v_new = [0, 0]
    p_new = [0, 0]

    for coord in range(2):
        v_new[coord] = alpha * V[particle][coord] + c1*r1*( P_star[particle][coord] - P[particle][coord] ) + c2*r2*(P_gb[coord] - P[particle][coord])

        # taking mod to make sure the position lies within the image
        p_new[coord] = (P[particle][coord] + v_new[coord]) % IMAGE_SIZE

    V[particle] = v_new
    P[particle] = p_new



    
def get_model_accuracy(particle_position):
    pass