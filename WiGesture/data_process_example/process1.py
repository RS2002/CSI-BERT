import numpy as np
import pickle
import os
import pandas as pd

root = "./data"
data = []
csi_vaid_subcarrier_index = range(0, 52)

def handle_complex_data(x, valid_indices):
    real_parts = []
    imag_parts = []
    for i in valid_indices:
        real_parts.append(x[i * 2])
        imag_parts.append(x[i * 2 - 1])
    return np.array(real_parts) + 1j * np.array(imag_parts)

people_id = 0
for dataset_dir in os.listdir(root):
    print(dataset_dir)
    dataset_path = os.path.join(root, dataset_dir)
    if not os.path.isdir(dataset_path):
        continue
    
    for static_dir in os.listdir(dataset_path):
        print(static_dir)
        static_path = os.path.join(dataset_path, static_dir)
        if not os.path.isdir(static_path):
            continue
        
        for id_dir in os.listdir(static_path):
            id_path = os.path.join(static_path, id_dir)
            if not os.path.isdir(id_path):
                continue
            
            print(f"Processing ID: {id_dir}")
            action_id = 0
            
            # Verifique se há arquivos CSV diretamente no diretório ID
            for item in os.listdir(id_path):
                item_path = os.path.join(id_path, item)
                
                # Se for um diretório, considere como um diretório de ação
                if os.path.isdir(item_path):
                    action_dir = item
                    print(f"  Action: {action_dir}")
                    
                    # Processe os arquivos CSV dentro do diretório de ação
                    for file in os.listdir(item_path):
                        if not file.endswith('.csv'):
                            continue
                        
                        file_path = os.path.join(item_path, file)
                        try:
                            df = pd.read_csv(file_path)
                            df.dropna(inplace=True)
                            df['data'] = df['data'].apply(lambda x: eval(x))
                            complex_data = df['data'].apply(lambda x: handle_complex_data(x, csi_vaid_subcarrier_index))
                            magnitude = complex_data.apply(lambda x: np.abs(x))
                            phase = complex_data.apply(lambda x: np.angle(x, deg=True))
                            time = np.array(df['timestamp'])
                            local_time = np.array(df['local_timestamp'])
                            
                            data.append({
                                'csi_time': time,
                                'csi_local_time': local_time,
                                'volunteer_name': id_dir,
                                'volunteer_id': people_id,
                                'action': action_dir,
                                'action_id': action_id,
                                'magnitude': np.array([np.array(a) for a in magnitude]),
                                'phase': np.array([np.array(a) for a in phase])
                            })
                        except Exception as e:
                            print(f"    Error processing {file_path}: {e}")
                    
                    action_id += 1
                
                # Se for um arquivo CSV, processe-o diretamente
                elif item.endswith('.csv'):
                    try:
                        df = pd.read_csv(item_path)
                        df.dropna(inplace=True)
                        df['data'] = df['data'].apply(lambda x: eval(x))
                        complex_data = df['data'].apply(lambda x: handle_complex_data(x, csi_vaid_subcarrier_index))
                        magnitude = complex_data.apply(lambda x: np.abs(x))
                        phase = complex_data.apply(lambda x: np.angle(x, deg=True))
                        time = np.array(df['timestamp'])
                        local_time = np.array(df['local_timestamp'])
                        
                        data.append({
                            'csi_time': time,
                            'csi_local_time': local_time,
                            'volunteer_name': id_dir,
                            'volunteer_id': people_id,
                            'action': 'unknown',
                            'action_id': 0,
                            'magnitude': np.array([np.array(a) for a in magnitude]),
                            'phase': np.array([np.array(a) for a in phase])
                        })
                    except Exception as e:
                        print(f"  Error processing {item_path}: {e}")
            
            people_id += 1

# Salve o dicionário global como um arquivo pickle
output_file = './csi_data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(data, f)
