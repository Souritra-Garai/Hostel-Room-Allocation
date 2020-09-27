import numpy as np

result_scores = np.load('Final_Room_Allocation_Scores.npy')
result_room_allocation = np.load('PSO_Final_Room_Allocations.npy')

room_capacity = np.genfromtxt('Room_Capacity_List.csv', dtype=np.int)
# print(result_scores)
# print(result_room_allocation.shape)

min_max = np.argmin(result_scores[:, 0])
min_sum = np.argmin(result_scores[:, 1])

# print(min_max, min_sum)

print('Room Allocation with Minimum Sum of Preferences')
print('Sum :', result_scores[min_sum, 1], '   Max :', result_scores[min_sum, 0])

k = 0
for i in range(40) :

    print('Room #', i+1, '  \t', result_room_allocation[min_sum, k]+1)
    k += 1
    
    if room_capacity[i] == 2 :

        print('Room #', i+1, '!!\t', result_room_allocation[min_sum, k]+1)
        k += 1

print('\nRoom Allocation with Minimum Max Preferences')
print('Sum :', result_scores[min_max, 1], '   Max :', result_scores[min_max, 0])
k = 0
for i in range(40) :

    print('Room #', i+1, '  \t', result_room_allocation[min_max, k]+1)
    k += 1
    
    if room_capacity[i] == 2 :

        print('Room #', i+1, '!!\t', result_room_allocation[min_max, k]+1)
        k += 1
