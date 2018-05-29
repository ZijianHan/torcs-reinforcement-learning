import h5py
filename = 'actormodel.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]
print(a_group_key)

# Get the data
data = list(f[a_group_key])
print(data)
