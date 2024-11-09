import pickle

# Replace 'data.pickle' with the path to your .pickle file
with open('allocine_dataset.pickle', 'rb') as file:
    data = pickle.load(file)

# Now, 'data' contains the deserialized Python object
print(data)
