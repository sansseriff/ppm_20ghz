import matplotlib.pyplot as plt

# create a sample plot with hist2d objects
fig, ax = plt.subplots()
ax.hist2d([1, 2, 3], [4, 5, 6])
# plt.show()

# check if there are QuadMesh objects in the current figure
for ax in fig.axes:

    quad_mesh = [
            child for child in ax.get_children() if isinstance(child, plt.matplotlib.collections.QuadMesh)
        ]

    # print(type(artist))
    # if isinstance(artist, plt.matplotlib.collections.QuadMesh):
    #     has_quadmesh = True
    #     break

    # for mesh in quad_mesh:
    #     mesh.set_cmap('plasma')

    # print(QuadMesh)

plt.savefig('test.png')



# print(has_quadmesh)