# Script to read amazon dataset

import scipy.io

def read():
    data = scipy.io.loadmat('datasets/Amazon-all.mat')
    network = data.get('Network')

    ### (11944, 11944)
    # print("Shape: ", network.shape)
    ### 8796784
    # print("Count non null elements: ", network.nnz)
    ### 133862352
    # print("Count null elements: ", 11944*11944 - network.nnz)

    ### <COOrdinate sparse matrix of dtype 'float64' with 8796784 stored elements and shape (11944, 11944)>
    ### ...
    # print(network)

    ### Coords	Values
    ### (157, 0)	1.0
    # print(network.toarray()[0][157])
    # print(network.toarray()[157][0])

    return network.toarray()

