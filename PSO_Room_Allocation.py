# Written by - Souritra Garai
# Contact - souritra.garai@iitgn.ac.in, sgarai65@gmail.com
# Data - 17th September 2020

from Particle_Swarm_Optimization import Single_Objective_PSO
from Room_Allocation_Methods import vectorized_calculate_fitness, allocate_students, calculate_preference_score_max, calculate_preference_score_sum, init_pool, destruct_pool
import numpy as np

num_iter = 10000
num_particles = 1000
num_runs = 25

problems = [Single_Objective_PSO() for i in range(num_runs)]

search_space_limits = [[0, i] for i in range(55, 0, -1)]

for problem in problems :

    alpha, c_g, c_p = np.random.rand(3)
    # print(type(alpha))

    problem.set_dimension_input(55)
    problem.set_fitness_function(vectorized_calculate_fitness)
    problem.set_inertia_factor(float(alpha))
    problem.set_learning_rates(float(c_g), float(c_p))
    problem.set_number_of_particles(num_particles)
    problem.set_search_space_limits(search_space_limits)

    # init_pool()

    problem.begin()

g_best_status = np.zeros((num_iter, num_runs))

try :

    for i in range(num_iter) :

        for j, problem in enumerate(problems) :
                
            problem.iterate()
            g_best_status[i, j] = - problem.get_global_best_value()

        if i % 10 == 0 :

            print(i*100 / num_iter, '%')

    print('100.00 %')
    # print('Stopped Internally')

except KeyboardInterrupt :

    print('Forced Stopped Externally')

finally :

    final_room_allocations = []
    final_room_allocation_scores = []

    for problem in problems :

        problem.stop_iterations()
        result = np.array(problem.get_global_best(), dtype=int)
        room_allocation = allocate_students(result)

        final_room_allocations.append(room_allocation)
        final_room_allocation_scores.append([calculate_preference_score_max(room_allocation), calculate_preference_score_sum(room_allocation)])  
        # print(room_allocation)
        # print(calculate_preference_score_max(room_allocation))
        #print(calculate_preference_score_sum(room_allocation))

    np.save('PSO_RUN_G_Status', g_best_status)
    np.save('PSO_Final_Room_Allocations', final_room_allocations)
    np.save('Final_Room_Allocation_Scores', final_room_allocation_scores)
    #np.save

    # destruct_pool()