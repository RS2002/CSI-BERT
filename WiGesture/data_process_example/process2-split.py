import numpy as np
import pickle
import copy

def get_time(s):
    s=s.split()[-1]
    s=s.split(":")
    h=float(s[0])
    m=float(s[1])
    t=float(s[2])
    total=h*3600+m*60+t
    return h,m,t,total

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
    start_time=None
    last_local=None
    current_magnitude=[]
    current_phase=[]
    current_timestamp=[]
    for i in range(len(csi_time)):
        _, _, _, current_time = get_time(csi_time[i])
        if start_time is None or current_time-start_time>gap:
            if start_time is not None:
                if len(current_magnitude)>=length:
                    current_magnitude=current_magnitude[:length]
                    current_phase=current_phase[:length]
                    current_timestamp=current_timestamp[:length]
                else:
                    add=length-len(current_magnitude)
                    delta=(current_timestamp[0]+length*loacl_gap-current_timestamp[-1])/add
                    for j in range(add):
                        current_magnitude.append(pad)
                        current_phase.append(pad)
                        current_timestamp.append(current_timestamp[-1]+delta)
                magnitudes.append(copy.deepcopy(current_magnitude))
                phases.append(copy.deepcopy(current_phase))
                timestamp.append(copy.deepcopy(current_timestamp))
                action_list.append(action)
                people_list.append(people)
                current_magnitude = []
                current_phase = []
                current_timestamp = []
            start_time=current_time
            last_local=local_time[i]
            current_magnitude.append(magnitude[i])
            current_phase.append(phase[i])
            current_timestamp.append(local_time[i])
        else:
            local = local_time[i]
            num=round((local-last_local-loacl_gap)/loacl_gap)
            if num>0:
                delta=(local-last_local)/(num+1)
                for j in range(num):
                    current_magnitude.append(pad)
                    current_phase.append(pad)
                    current_timestamp.append(current_timestamp[-1] + delta)
            current_magnitude.append(magnitude[i])
            current_phase.append(phase[i])
            current_timestamp.append(local_time[i])
            last_local=local


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
np.save("./magnitude.npy", np.array(magnitudes))
np.save("./phase.npy", np.array(phases))
np.save("./action.npy", np.array(action_list))
np.save("./people.npy", np.array(people_list))
np.save("./timestamp.npy", np.array(timestamp))
