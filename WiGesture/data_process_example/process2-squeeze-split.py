import numpy as np
import pickle
import copy


gap=1
length=gap*100
pad=[-1000]*52
action_list=[]
people_list=[]
timestamp=[]
magnitudes=[]
phases=[]
loacl_gap=10000


with open("./csi_data.pkl", 'rb') as f:
    csi = pickle.load(f)

for data in csi:
    csi_time=data['csi_time']
    local_time=data['csi_local_time']
    magnitude=data['magnitude']
    phase=data['phase']
    people=data['volunteer_id']
    action=data['action_id']

    index=0
    while index<len(magnitude)-length:
        current_magnitude=magnitude[index:index+length]
        current_phase=phase[index:index+length]
        current_timestamp=local_time[index:index+length]
        index+=(length+gap-1)
        magnitudes.append(copy.deepcopy(current_magnitude))
        phases.append(copy.deepcopy(current_phase))
        timestamp.append(copy.deepcopy(current_timestamp))
        action_list.append(action)
        people_list.append(people)

action_list=np.array(action_list)
people_list=np.array(people_list)
timestamp=np.array(timestamp)
magnitudes=np.array(magnitudes)
phases=np.array(phases)
print(action_list.shape)
print(people_list.shape)
print(timestamp.shape)
print(magnitudes.shape)
print(phases.shape)
np.save("./squeeze_data/magnitude.npy", np.array(magnitudes))
np.save("./squeeze_data/phase.npy", np.array(phases))
np.save("./squeeze_data/action.npy", np.array(action_list))
np.save("./squeeze_data/people.npy", np.array(people_list))
np.save("./squeeze_data/timestamp.npy", np.array(timestamp))
