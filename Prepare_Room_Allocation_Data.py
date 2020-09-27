# Written by - Souritra Garai
# Contact - souritra.garai@iitgn.ac.in, sgarai65@gmail.com
# Data - 15th September 2020

import numpy as np

room_preferences_data = np.genfromtxt('Room_Preference_Data.csv', dtype=np.int)
# print('Number of Students :', room_preferences_data.shape[0])

room_capacity = np.genfromtxt('Room_Capacity_List.csv', dtype=np.int)
total_capacity = np.sum(room_capacity)
# print('Total Capacity of all Rooms :', total_capacity)

room_id = []
double_occupancy_rooms = []

k = 0

for i in range(room_capacity.shape[0]) :

    for j in range(room_capacity[i]) :

        room_id.append(i)

    if room_capacity[i] == 2 :

        double_occupancy_rooms.append([k, k+1])
        k += 1

    k += 1

# print(room_id)
# print(double_occupancy_rooms)
# print(len(room_id))

room_preference = np.zeros((room_preferences_data.shape[0], total_capacity))

for i in range(room_preferences_data.shape[0]) :

    room_preference[i] = room_preferences_data[i, room_id]

# print(room_preference)
# print(room_preference.shape)

np.save('Room_Preference_Matrix', room_preference)
np.save('Double_Occupancy_Rooms', double_occupancy_rooms)

