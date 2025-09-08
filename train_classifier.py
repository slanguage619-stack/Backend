import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

# Check the data structure and lengths
print(f"Total samples: {len(data_dict['data'])}")
print(f"Sample lengths: {[len(sample) for sample in data_dict['data'][:10]]}")

# Filter out samples with inconsistent lengths
# MediaPipe hand landmarks should have 21 points * 2 coordinates = 42 features
expected_length = 42
filtered_data = []
filtered_labels = []

for i, (sample, label) in enumerate(zip(data_dict['data'], data_dict['labels'])):
    if len(sample) == expected_length:
        filtered_data.append(sample)
        filtered_labels.append(label)
    else:
        print(f"Skipping sample {i} with length {len(sample)} (expected {expected_length})")

print(f"Filtered samples: {len(filtered_data)} out of {len(data_dict['data'])}")

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Check if we have enough data
if len(data) < 10:
    print("Error: Not enough valid samples to train the model!")
    exit()

print(f"Training with {len(data)} samples, shape: {data.shape}")

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
