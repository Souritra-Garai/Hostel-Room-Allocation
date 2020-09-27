# Hostel-Room-Allocation

For description of the project watch the [youtube video](https://youtu.be/g1qbxnmESRs)

Sequence of Running the code -

1. **Run Prepare_Room_Allocation_Data.py :**
It requires Room_Capacity_List.csv and Room_Preference_Data.csv.
It generates Room_Preference_Matrix.npy annd Double_Occupancy_Rooms.npy

2. **Run PSO_Room_Allocation.py :**
It is dependent on Particle_Swarm_Optimization.py and Room_Allocation_Methods.py.
Room_Allocation_Methods.py requires the data - Room_Preference_Matrix.npy annd Double_Occupancy_Rooms.npy.
PSO_Room_Allocation.py will generate PSO_RUN_G_Status.npy, PSO_Final_Room_Allocations.npy and Final_Room_Allocation_Scores.npy.

3. **Run Room_Allocation_PSO_Visualization.py :**
It requires PSO_RUN_G_Status.npy.
It generates a plot of evolution of the global best values with iterations saves it as PSO_Original.mp4.

4. **Run Room_Allocation_Results.py :**
It requires PSO_Final_Room_Allocations.npy, Final_Room_Allocation_Scores.npy and Room_Capacity_List.csv.
It displays the room allocation for the minimum sum of preferences and minimum upper bound on individual preferences.
