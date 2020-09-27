# Written by - Souritra Garai
# Contact - souritra.garai@iitgn.ac.in, sgarai65@gmail.com
# Data - 15th September 2020

import os
import numpy as np
import multiprocessing as mp

from numba import jit

pool = None

def init_pool() :

    global pool
    pool = mp.Pool(mp.cpu_count())

    pass

def destruct_pool() :

    global pool
    pool.close()

    pass

room_preferences = np.load(os.path.join('npy Files','Room_Preference_Matrix.npy'))
num_students, num_rooms = room_preferences.shape

double_occupancy_rooms = np.load(os.path.join('npy Files','Double_Occupancy_Rooms.npy'))

# allocation_list = np.zeros(num_rooms, dtype=np.int)

@jit(nopython=True)
def allocate_students(vector) :

    '''
    for i, x in enumerate(vector) :

        # print('x', x)
        allocation_list[i] = x + np.count_nonzero(vector[:i] <= x)

    allocation_list[-1] = np.count_nonzero(vector == 0)

    print('vector')
    print(vector)
    print(allocation_list)
    print(np.unique(allocation_list, return_counts=True))    
    
    '''
    students = [i for i in range(num_students)]
    # num_unallocated_students = num_students

    # print('vector', vector)
    allocation_list = [students.pop(x) for x in vector]
    allocation_list.append(students.pop(0))

    '''
    for i, x in enumerate(vector) :

        # print('x', x)
        allocation_list[i] = students.pop(x % num_unallocated_students)
        num_unallocated_students -= 1

    allocation_list[num_rooms-1] = students.pop(0)
    '''
    # print('vector')
    # print(vector)
    # print(allocation_list)
    # print(np.unique(allocation_list, return_counts=True))    

    # print(allocation_list)

    return allocation_list


def calculate_preference_score_sum(allocated_students) :

    # gender_check = allocated_students[double_occupancy_rooms] > 28

    students, allocated_rooms = np.unique(allocated_students, return_inverse=True)

    # if np.logical_xor(gender_check[:, 0], gender_check[:, 1]).any() :

        # print('Girls and Boys in the same room!!')

    return np.sum(room_preferences[students, allocated_rooms])


def calculate_preference_score_max(allocated_students) :

    students, allocated_rooms = np.unique(allocated_students, return_inverse=True)

    return np.amax(room_preferences[students, allocated_rooms])


def calculate_fitness(vector) :

    return - calculate_preference_score_sum(allocate_students(np.array(vector, dtype=int)))


def vectorized_calculate_fitness(matrix) :

    global pool

    # print('got called')

    if type(pool) != type(None) :

        # print('v')

        retval = np.array(pool.map(calculate_fitness, matrix))
    
    else :

        retval = np.zeros(matrix.shape[0])

        for i, vector in enumerate(matrix) :

            # print('v', vector)

            retval[i] = calculate_fitness(vector)

    return retval

if __name__ == "__main__":

    '''
    vector = np.random.randint(0, num_rooms, num_students-1)
    v2 = allocate_rooms(vector)
    print(vector)
    print(v2)
    print(calculate_fitness(v2))
    '''

    # init_pool()

    for i in range(1) :

        vector1 = np.array(np.random.randint(0, num_rooms, num_students-1), dtype=float) % np.arange(num_students-1, 0, -1)
        vector2 = np.random.randint(0, num_rooms, num_students-1) % np.arange(num_students-1, 0, -1)
        # print(vector1, vector2)

        # print(i+1, calculate_fitness(vector1), calculate_fitness(vector2))
        print(i+1, vectorized_calculate_fitness(np.array([vector1, vector2])))

    # destruct_pool()
