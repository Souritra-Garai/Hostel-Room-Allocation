# Written by - Souritra Garai
# Contact - souritra.garai@iitgn.ac.in, sgarai65@gmail.com

# Definition of a class for using Particle Swarm Optimization
# Uses numpy to optimize iterations and other linear algebra operations
# Make use of Numpy Universal Functions to define the fitness function

import numpy as np

# defination of class to solve single objective optimization problem
# using particle swarm method
class Single_Objective_PSO :

    def __init__(self) :

        # Dimension of input
        self.__n = None

        # Number of particles
        self.__num = None

        # Personal best position
        self.__p_best_pos = None
        
        # Personal best fitness value
        self.__p_best_val = None

        # Learning rate for personal best
        self.__c_p = None

        # Global best position
        self.__g_best_pos = None

        # Global best value
        self.__g_best_val = None

        # Learning rate from global best
        self.__c_g = None

        # Inertia factor
        self.__alpha = None

        # Particles' position
        self.__r = None

        # Limits particles' position
        self.__lim = None

        # Particles' velocity
        self.__v = None

        # Particles' fitness
        self.__f = None

        # Function to evaluate fitness of particles
        self.__fitness_function = None
    
        # Flag if ready to iterate
        self.__not_ready_2_iterate = True

        # Variables keeping track on number of iterations
        # self.__num_iterations = None

    def set_dimension_input(self, n) :
        ''' Sets the dimension of the seach space to n  '''

        if self.__not_ready_2_iterate == False :

            raise RuntimeError('Running iterations, cannot update now, use stop_iterations() first!!')

        if type(n)!=int :

            raise TypeError('Can accept integer only!!\n')

        if n < 1 :

            raise ValueError('At least 1 dimension is required!!\n')

        self.__n = n

        pass

    def set_number_of_particles(self, num) :
        ''' Sets the number of particles to be used / population size
            for searching the search space to num   '''

        if self.__not_ready_2_iterate == False :

            raise RuntimeError('Running iterations, cannot update now, use stop_iterations() first!!')

        if type(num)!=int :

            raise TypeError('Can accept integer only!!\n')

        if num < 1 :

            raise ValueError('At least 1 particle is required!!\n')

        self.__num = num
        pass

    def set_learning_rates(self, c_g, c_p) :
        ''' Sets the learning rates from global best and
            personal bests -
            - c_g : learning rate for global best
            - c_p : learning rate for personal best '''

        if self.__not_ready_2_iterate == False :

            raise RuntimeError('Running iterations, cannot update now, use stop_iterations() first!!')

        if type(c_p)!=float :

            raise TypeError('c_p can be float only!!\n')

        if type(c_g)!=float :

            raise TypeError('c_g can be float only!!\n')

        self.__c_g = c_g
        self.__c_p = c_p
        pass

    def set_inertia_factor(self, alpha) :
        ''' Set the inertia factor alpha
            weight for previous velocity  '''

        if self.__not_ready_2_iterate == False :

            raise RuntimeError('Running iterations, cannot update now, use stop_iterations() first!!')

        if type(alpha)!=float :

            raise TypeError('c_g can be float only!!\n')

        self.__alpha = alpha
        pass

    def set_search_space_limits(self, array) :
        ''' Sets the lower and upper limtis on the search space
            Array should contain n pairs of float type numbers  '''

        if self.__not_ready_2_iterate == False :

            raise RuntimeError('Running iterations, cannot update now, use stop_iterations() first!!')
        
        my_array = np.array(array, dtype=np.float).reshape(-1, 2)
        
        if self.__n != None and self.__n != np.shape(my_array)[0] :

            raise ValueError('Dimension mismatch!!\n')

        elif self.__n == None :

            self.__n = np.shape(my_array)[0]

        self.__lim = my_array
        pass

    def __init_particles(self) :
        ''' Initialize the particles' position and velocities   '''

        # randomly setting position of particles in the search space
        # particle_position = search_space_min + random_number * (search_space_max - search_space_min)
        self.__r = self.__lim[:, 0] + np.random.rand(self.__num, self.__n)*(self.__lim[:, 1] - self.__lim[:, 0])
        
        # randomly setting velocity
        # particle_velocity = random_number * (search_space_max - search_space_min)
        self.__v = np.random.rand(self.__num, self.__n)*(self.__lim[:, 1] - self.__lim[:, 0])
        
        # evaluating the fitness
        self.__f = self.__fitness_function(self.__r)
        pass

    def check_if_initialized(self) :

        variables = ''

        if type(self.__lim) == type(None) :

            variables += 'limits on search space, '

        if self.__n == None :

            variables += 'dimension of search space, '

        if self.__num == None :

            variables += 'number of particles to be used, '

        if self.__alpha == None :

            variables += 'inertia factor alpha, '

        if self.__c_g == None :

            variables += 'learning rate from global best, '

        if self.__c_p == None :

            variables += 'learning rate from personal best, '

        if self.__fitness_function == None :

            variables += 'function to evaluate fitness, '

        if len(variables) > 0 :

            raise RuntimeError(variables[:-2] + ' are not yet initialised!!')

        pass

    def set_fitness_function(self, function) :
        ''' Sets the fitness function used to evaluate the fitness 
            at a position. Function should accept the 
            particles' position matrix as input '''

        if self.__not_ready_2_iterate == False :

            raise RuntimeError('Running iterations, cannot update now, use stop_iterations() first!!')
        
        if not callable(function) :

            raise TypeError('function has to be callable!!\n')

        self.__fitness_function = function
        pass

    def begin(self) :
        ''' Perform the beginning steps of the iterations   '''

        if self.__not_ready_2_iterate == False :

            raise RuntimeError('Running iterations, cannot update now, use stop_iterations() first!!')
        
        self.check_if_initialized()

        self.__init_particles()

        # set the initial positions as personal best positions
        self.__p_best_pos = np.copy(self.__r)
        self.__p_best_val = np.copy(self.__f)

        # find the index of the particle with highest fitness and
        # set it as global best
        i = np.argmax(self.__f)
        self.__g_best_pos = np.copy(self.__r[i])
        self.__g_best_val = np.copy(self.__f[i])

        self.__not_ready_2_iterate = False
        # self.__num_iterations = 0
        pass

    def __move_particles(self) :
        ''' Moves the particles by the velocity in __v
            and if any particle moves outside the search space
            assigns a random position to them   '''

        self.__r += self.__v

        # indices of rows where particles go beyond search space
        i = ((self.__r >= self.__lim[:, 1]) + (self.__r <= self.__lim[:, 0])).any(axis=1).nonzero()[0]

        # set random position to those particles
        self.__r[i,:] = self.__lim[:, 0] + np.random.rand(i.shape[0], self.__n)*(self.__lim[:, 1] - self.__lim[:, 0])

        pass      

    def __update_best(self) :
        ''' Updates the personal best and global best variables '''

        # indices of particles whose current fitness value is greater than those of the previous p_best
        i = (self.__f > self.__p_best_val).nonzero()[0]

        # if i is not empty
        if i.shape[0] > 0 :

            # change the personal best for the i particular particles
            self.__p_best_val[i] = np.copy(self.__f[i])
            self.__p_best_pos[i,:] = np.copy(self.__r[i,:])
            
            # find the index of the global best candidate among the changed personal best particles
            i = i[np.argmax(self.__f[i])]
            # update global best if smaller than particle with highest fitness 
            if self.__f[i] > self.__g_best_val :

                # print(self.__f[i], '>', self.__g_best_val)

                self.__g_best_val = np.copy(self.__f[i])
                self.__g_best_pos = np.copy(self.__r[i,:])

        pass

    def __update_velocity(self) :
        ''' Updates the velocity __v using learning and inertial factors    '''
  
        self.__v = (        self.__alpha * self.__v 
                        +   self.__c_g * np.random.rand(self.__num, 1) * (self.__g_best_pos - self.__r)
                        +   self.__c_p * np.random.rand(self.__num, 1) * (self.__p_best_pos - self.__r)    )

        pass

    def iterate(self) :
        ''' Take one step in the iterative process  '''

        if self.__not_ready_2_iterate :

            raise RuntimeError('Not yet ready to iterate, first call begin()!!')

        self.__move_particles()

        self.__f = self.__fitness_function(self.__r)
        
        self.__update_best()
        
        self.__update_velocity()
        
        # self.__num_iterations += 1
        pass

    def stop_iterations(self) :
        ''' Let optimizer know that iterations are over.
            You may edit values and restart iterations using begin()    '''

        self.__not_ready_2_iterate = True

        pass

    def get_global_best(self) :

        # print('Global Best Vector :', self.__g_best_pos)
        # print('Fitness Value :', self.__g_best_val)

        return np.copy(self.__g_best_pos)

    def get_global_best_value(self) :

        return np.copy(self.__g_best_val)

    def get_particle_positions(self) :
        ''' Returns a copy of the (num_particles) X (dimension of input) matrix
            holding the positions of the particles at present   '''

        return np.copy(self.__r)

if __name__ == "__main__":

    problem = Single_Objective_PSO()
    
    # a.set_dimension_input(1)
    def myfunc(x) :

        return np.sin(x)

    problem.set_fitness_function(myfunc)
    problem.set_inertia_factor(0.85)
    problem.set_learning_rates(0.6, 0.8)
    problem.set_number_of_particles(100)
    problem.set_search_space_limits(np.array([[0, np.pi]]))
    
    problem.begin()
    
    for i in range(1000) :
        problem.iterate()
        # print('Global Best Vector :', a.get_global_best())
        # print('Fitness Value :', -np.sin(a.get_global_best()))
    
    problem.stop_iterations()

    print('Global Best Vector :', problem.get_global_best())
    print('Fitness Value :', myfunc(problem.get_global_best()))