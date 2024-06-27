import numpy as np
import pickle


result=[]

pad=[-1000]*52
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

    last_local=None
    current_magnitude=[]
    current_phase=[]
    current_timestamp=[]
    for i in range(len(csi_time)):
        if last_local is None:
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

    # print(len(current_magnitude))
    result.append({
        'time': np.array(current_timestamp),
        'people': people,
        'action': action,
        'magnitude': np.array(current_magnitude),
        'phase': np.array(current_phase)
    })

output_file = './data_sequence.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(result, f)

