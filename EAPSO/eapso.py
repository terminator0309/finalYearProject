import numpy as np
import random
import cv2 as cv
from keras.models import load_model

# image to be used for evasion attack
IMAGE_NAME="00000_00000_00000.png"
IMAGE=[]

#image size
IMAGE_SIZE = 32

# number of particles in the swarm
M = 10

# iteration counter
T = 10

# positions of particles
P = [[random.randint(0, IMAGE_SIZE-1), random.randint(0, IMAGE_SIZE-1)] for _ in range(M)]

# best position of particles
P_star = P

# best position of particles acc
Prob_star = [ 0 for _ in range(M) ]

# best position globally
P_gb = [0,0]

# best position globally accuracy
Prob_gb = 100

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


def get_sample_image():
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

    min_probability=1
    P_gb=[]

    for particle in range(M):

        x = P[particle][0]
        y = P[particle][1]

        probability = sign_predict(get_image_with_perturbation(x, y))
        print(f"Particle: {particle} [{x}, {y}], probability: {min_probability}", end="\n\n")
        if probability < min_probability:
            P_gb = P[particle]
            min_probability = probability

    Prob_gb = min_probability
    print("Initialized P global with probability: ", Prob_gb)
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


def sign_predict(image):
    model = load_model('./keras_model/')
    #image = np.array(image, dtype=np.float32)
    image = image/255
    image = np.reshape(image, (1, 32, 32))
    x = image.astype(np.float32)
    prediction = model.predict(x)
    #print(prediction)
    confidence = np.max(prediction)
    return confidence


def generate_evasioned_sample(P_gb, Prob_gb):
    for i in range(T):
        for particle in range(M):
            print("Iteration: ", i)
            print("Particle: ", particle)
            update_attributes(particle)

            x = P[particle][0]
            y = P[particle][1]

            model_probability_for_particle_new_position = sign_predict(get_image_with_perturbation(x, y))
            model_probability_for_particle_best_position = Prob_star[particle]

            model_probability_for_particle_global_position = Prob_gb

            if model_probability_for_particle_new_position < model_probability_for_particle_best_position:
                P_star[particle] = P[particle]
                Prob_star[particle] = model_probability_for_particle_new_position

            if model_probability_for_particle_new_position < model_probability_for_particle_global_position:
                P_gb = P[particle]
                Prob_gb = model_probability_for_particle_new_position

        print(f'After iteration : ({i}), probability: {Prob_gb}')

    return P_gb, Prob_gb



IMAGE = get_sample_image()

P_gb = initialize_P_global()

P_gb, Prob_gb = generate_evasioned_sample(P_gb, Prob_gb)

print("Position : ", P_gb)
print("Probability : ", Prob_gb)

# 0.99935156
for _ in range(1):
    print("Without perturbration probability: ", sign_predict(IMAGE))

countWhite=0
for row in IMAGE:
    for pixel in row:
        #print(pixel)
        if pixel == 255:
            countWhite+=1

print("white check",countWhite)



# cv.imshow('evasion attack', get_image_with_perturbation(P_gb[0], P_gb[1]))
# cv.waitKey(0)
# cv.destroyAllWindows()