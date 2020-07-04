import numpy as np
import scipy.sparse as sp
import gcn.utils as gut


class BasicWeighting:
    def __init__(self, w_id):
        self.description = "All edges are weighted as 1"
        self.id = w_id
        self.weights = []

    def weights_for(self, idx1, idx2, args):
        self.weights.append(1)

    def post_process(self, args=None):
        self.weights = np.asarray(self.weights, dtype=np.float32)
        num_nodes = args["num_nodes"]
        w1 = sp.coo_matrix((self.weights, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))
        self.weights = w1

    def get_weights(self):
        return self.weights

    def get_id(self):
        return self.id

    @property
    def get_description(self):
        return self.description


class Weighting1(BasicWeighting):
    def __init__(self, w_id):
        super(Weighting1, self).__init__(w_id=w_id)
        self.description = "l*div + e(int) + e(pos)"
        self.weights1 = []
        self.weights2 = []
        self.weights3 = []

    def weights_for(self, idx1, idx2, args):
        prob1 = args["probability"][idx1]
        prob2 = args["probability"][idx2]
        int1 = args["volume"][idx1]
        int2 = args["volume"][idx2]
        ny, nx, nz = args["volume"].shape
        dim_array = np.array([ny, nx, nz], dtype=np.float32)
        pos1 = np.array(idx1,dtype=np.float32) / dim_array
        pos2 = np.array(idx2, dtype=np.float32) / dim_array
#       Computing the weight
        int_diff = int1 - int2
        pos_diff = pos1 - pos2
        intensity = np.sum(int_diff * int_diff)
        space = np.sum(pos_diff * pos_diff)
        p = prob1 - prob2
        delta = 1.0e-15
        lambd = p * (np.log2(prob1 / (prob2 + delta) + delta) - np.log2((1 - prob1) / ((1 - prob2) + delta) + delta))
        self.weights1.append(lambd)
        self.weights2.append(intensity)
        self.weights3.append(space)

    def post_process(self, args=None):
        self.weights1 = np.asarray(self.weights1, dtype=np.float32)
        self.weights2 = np.asarray(self.weights2, dtype=np.float32)
        self.weights3 = np.asarray(self.weights3, dtype=np.float32)
        num_nodes = args["num_nodes"]
        ne = float(self.weights1.shape[0])
        muw2 = self.weights2.sum() / ne
        muw3 = self.weights3.sum() / ne

        sig2 = 2 * np.sum((self.weights2 - muw2) ** 2) / ne
        sig3 = 2 * np.sum((self.weights3 - muw3) ** 2) / ne

        self.weights2 = np.exp(-self.weights2 / sig2)
        self.weights3 = np.exp(-self.weights3 / sig3)

        w1 = sp.coo_matrix((self.weights1, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))
        w2 = sp.coo_matrix((self.weights2, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))
        w3 = sp.coo_matrix((self.weights3, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))

        self.weights = 0.5 * w1 + w2 + w3


def get_weighting_func(w_id):
    if w_id == 0:
        return BasicWeighting(w_id)
    if w_id == 1:
        return Weighting1(w_id=1)
    return None



