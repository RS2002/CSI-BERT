import numpy as np
import pickle
import os
import pandas as pd

root="./data"
data=[]
csi_vaid_subcarrier_index = range(0, 52)

def handle_complex_data(x, valid_indices):
    real_parts = []
    imag_parts = []
    for i in valid_indices:
        real_parts.append(x[i * 2])
        imag_parts.append(x[i * 2 - 1])
    return np.array(real_parts) + 1j * np.array(imag_parts)



people_id=0
for people in os.listdir(root):
    print(people)
    action_id=0
    path_people=os.path.join(root,people)
    for action in os.listdir(path_people):
        count=0
        print(action)
        path_action=os.path.join(path_people,action)
        for file in os.listdir(path_action):
            count+=1
            path=os.path.join(path_action,file)
            df = pd.read_csv(path)
            df.dropna(inplace=True)
            df['data'] = df['data'].apply(lambda x: eval(x))
            complex_data = df['data'].apply(lambda x: handle_complex_data(x, csi_vaid_subcarrier_index))
            magnitude = complex_data.apply(lambda x: np.abs(x))
            phase = complex_data.apply(lambda x: np.angle(x, deg=True))
            time = np.array(df['timestamp'])
            local_time = np.array(df['local_timestamp'])
            data.append({
                'csi_time':time,
                'csi_local_time':local_time,
                'volunteer_name': people,
                'volunteer_id': people_id,
                'action': action,
                'action_id': action_id,
                'magnitude': np.array([np.array(a) for a in magnitude]),
                'phase': np.array([np.array(a) for a in phase])
            })
            # if count==4:
            #     break
        action_id+=1
    people_id+=1

# 保存全局字典为一个pickle文件
output_file = './csi_data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(data, f)