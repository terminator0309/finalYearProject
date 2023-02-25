import random

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