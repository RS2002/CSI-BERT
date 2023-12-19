from torch.utils.data import Dataset
import numpy as np

class CSI_dataset(Dataset):
    def __init__(self, magnitudes, phases=None, timestamp=None, label_action=None, label_people=None):
        super().__init__()
        self.magnitudes = magnitudes
        self.phases = phases
        self.timestamp=timestamp
        self.label_action = label_action
        self.label_people = label_people
        self.num=self.magnitudes.shape[0]
        if self.phases is None:
            self.timestamp = [-1] * len(self.num)
        if self.timestamp is None:
            self.timestamp = [-1] * len(self.num)
        if self.label_action is None:
            self.label_action = [-1] * len(self.num)
        if self.label_people is None:
            self.label_people = [-1] * len(self.num)


    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.magnitudes[index],self.phases[index], self.label_action[index], self.label_people[index], self.timestamp[index]


def load_zero_people(test_people_list):
    magnitude=np.load("./data/magnitude.npy").astype(np.float32)
    phase=np.load("./data/phase.npy").astype(np.float32)
    timestamp=np.load("./data/timestamp.npy").astype(np.float32)
    people=np.load("./data/people.npy").astype(np.int64)
    action=np.load("./data/action.npy").astype(np.int64)
    a = np.zeros_like(people)
    b = np.zeros_like(people)
    for i in range(people.shape[0]):
        a[i]=(people[i] not in test_people_list)
        b[i]=not a[i]
    a=a.astype(bool)
    b=b.astype(bool)
    return CSI_dataset(magnitude[a], phase[a], timestamp[a], action[a], people[a]),CSI_dataset(magnitude[b], phase[b], timestamp[a], action[b], people[b])

def load_all(magnitude_path="./data/magnitude.npy",phase_path="./data/magnitude.npy"):
    magnitude=np.load(magnitude_path).astype(np.float32)
    phase=np.load(phase_path).astype(np.float32)
    phase[np.isnan(phase)] = -1000
    timestamp=np.load("./data/timestamp.npy").astype(np.float32)
    people=np.load("./data/people.npy").astype(np.int64)
    action=np.load("./data/action.npy").astype(np.int64)
    return CSI_dataset(magnitude, phase, timestamp, action, people)

def load_data(magnitude_path="./data/magnitude.npy",train_prop=None):
    magnitude=np.load(magnitude_path).astype(np.float32)
    phase=np.load("./data/phase.npy").astype(np.float32)
    timestamp=np.load("./data/timestamp.npy").astype(np.float32)
    people=np.load("./data/people.npy").astype(np.int64)
    action=np.load("./data/action.npy").astype(np.int64)
    if train_prop is None:
        return CSI_dataset(magnitude, phase, timestamp, action, people)
    else:
        a = np.zeros_like(people)
        num=[]
        current_num=0
        current_action=None
        for i in range(action.shape[0]):
            if action[i]==current_action:
                current_num+=1
            else:
                current_action = action[i]
                if current_action is None:
                    current_num+=1
                else:
                    num.append(current_num)
                    current_num=0
        current_num=0
        for i in range(len(num)):
            a[current_num:current_num+int(num[i]*train_prop)]=1
            current_num+=num[i]
        b=1-a
        a = a.astype(bool)
        b = b.astype(bool)
        return CSI_dataset(magnitude[a], phase[a], timestamp[a], action[a], people[a]), CSI_dataset(magnitude[b], phase[b], timestamp[b], action[b], people[b])