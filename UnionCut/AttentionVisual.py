# Created by Ziling Wu
# Supervised by Dr Armaghan Moemeni
# Last Updated: December 4th, 2023
import torch
from torchvision import transforms as pth_transforms
import math
import cv2
import numpy as np
from tqdm import tqdm
import os
from CRF.crf import densecrf
from sklearn.cluster import MeanShift, estimate_bandwidth
import torch.nn.functional as F

def relative2absolute(index):
    """
    transfer indices of matrix to array
    index: (i, j)
    """
    return index[0] * 28 + index[1]

def absolute2relative(index):
    """
    transfer indices of array to matrix
    indices: np.array([i,....])
    """
    j_s = index % 28
    i_s = (index - j_s) // 28
    return i_s, j_s


class Edge:
    """
    a class for edge
    """

    def __init__(self, _from, to):
        self._from = _from  # vertex the edge starts from
        self.to = to  # vertex the edge ends to
        self.weight = 0  # the weight of the edge, initially 0
        # parameters for max flow
        self.capacity = 0  # capacity of this link

    def calculateWeight(self, beta, square_distance_matrix, fore_seed_index=None, back_indices=None, K=None):
        """
        calculate the weight of the edge
        :param beta: weight for n-link cost
        :param square_distance_matrix: squared distance matrix for each feature point
        :param fore_seed_index: absolute index of the foreground seed
        :param back_indices: absolute indices of the background points
        :param K: weight for t-link ground truth
        :return:
        """
        # calculate weight for n-link
        if self._from.isTerminal is not True and self.to.isTerminal is not True:

            self.weight = 100 * math.exp(-square_distance_matrix[relative2absolute((self._from.row, self._from.col)), \
                relative2absolute((self.to.row, self.to.col))] / beta) / \
                          math.sqrt((self._from.row - self.to.row) ** 2 + (self._from.col - self.to.col) ** 2)
        # calculate weight for t-link
        elif self._from.isTerminal:
            # obtain the absolute index of the node
            abs_index = relative2absolute((self.to.row, self.to.col))
            # calculate the distance from the fore seed to the point
            fore_distance = np.sqrt(square_distance_matrix[abs_index, fore_seed_index])
            # find the minimum distance from background points to the point
            min_back_distance = np.sqrt(np.min(square_distance_matrix[abs_index, back_indices]))
            # edge from source to pixel
            if fore_distance == 0:
                # the pixel is assigned to foreground
                self.weight = K
            elif min_back_distance == 0:
                # the pixel is assigned to background
                self.weight = 0
            else:
                # the pixel has not been assigned to a label
                self.weight = - math.log(np.exp(fore_distance) / (np.exp(fore_distance) + np.exp(min_back_distance)))
        elif self.to.isTerminal:
            # obtain the absolute index of the node
            abs_index = relative2absolute((self._from.row, self._from.col))
            # calculate the distance from the fore seed to the point
            fore_distance = np.sqrt(square_distance_matrix[abs_index, fore_seed_index])
            # find the minimum distance from background points to the point
            min_back_distance = np.sqrt(np.min(square_distance_matrix[abs_index, back_indices]))
            # edge from source to pixel
            if fore_distance == 0:
                # the pixel is assigned to foreground
                self.weight = 0
            elif min_back_distance == 0:
                # the pixel is assigned to background
                self.weight = K
            else:
                # the pixel has not been assigned to a label
                self.weight = - math.log(np.exp(min_back_distance) / (np.exp(fore_distance) + np.exp(min_back_distance)))
        # update capacity
        self.capacity = self.weight


class Vertex:
    """
    a class for a pixel, which is represented as a vertex in the graph
    """

    def __init__(self, row=None, col=None, isTerminal=False):
        self.isTerminal = isTerminal  # terminals indicator
        # initialize the coordinate of the pixel
        self.row = row  # the index of row
        self.col = col  # the index of col
        self.edges = {}  # out link edges
        # parameters for max flow
        self.tree = None
        self.parent = None
        self.isActive = False

    def return_degree(self):
        """
        return the total weights of edges connected to the node
        """
        degree = 0
        for edge in self.edges.values():
            degree += edge.weight
        return degree


class GraphCut:
    """
    implementation of graph cut
    """

    def __init__(self, square_distance_matrix, fore_seed_index, back_indices):
        """
        :param square_distance_matrix: squared distance matrix for each feature point
        :param fore_seed_index: absolute index of the foreground seed
        :param back_indices: absolute indices ot the background
        :return:
        """
        self.square_distance_matrix = square_distance_matrix
        self.fore_seed_index = fore_seed_index
        self.back_indices = back_indices
        # obtain the size of the feature map
        self.row = 28
        self.col = 28
        # initialize weight for n-links
        self.beta = 0
        # initialize weight fot t-link ground truth
        self.K = 0
        # a list storing all vertices
        self.vertices = []
        # initialize edges
        self.edges = []
        # source node that indicates foreground
        self.source = Vertex(isTerminal=True)
        # sink node that indicates background
        self.sink = Vertex(isTerminal=True)
        # initialize graph
        self._initialize_nlinks()
        self._initialize_tlinks()

    def refresh_graph(self, new_fore_seed_index, new_back_indices):
        """
        refresh the graph to prepare for the next segmentation
        new_fore_seed_index: fore seed index for the next segmentation
        new_back_indices: background indices for the next segmentation
        """
        # reinitialize source and sink
        # source node that indicates foreground
        self.source = Vertex(isTerminal=True)
        # sink node that indicates background
        self.sink = Vertex(isTerminal=True)
        # update fore and back indices
        self.fore_seed_index = new_fore_seed_index
        self.back_indices = new_back_indices
        # reinitialize tlinks
        self._initialize_tlinks()
        # refresh max flow
        for i in range(self.row):
            for j in range(self.col):
                self.vertices[i][j].tree = None
                self.vertices[i][j].parent = None
                self.vertices[i][j].isActive = False

    def _initialize_nlinks(self):
        # transfer each pixel into vertex
        self.vertices = [[Vertex(i, j) for j in range(self.col)] for i in range(self.row)]
        # initialize n-link edges
        sigmas = []  # list for calculating beta
        for i in range(self.row):
            for j in range(self.col):
                if i - 1 >= 0 and j - 1 >= 0:
                    # create top left edge
                    topLeft = Edge(self.vertices[i][j], self.vertices[i - 1][j - 1])
                    # put the edge into the graph
                    self.edges.append(topLeft)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i - 1, j - 1)] = topLeft
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j)), relative2absolute((i-1, j-1))])
                if i - 1 >= 0:
                    # create top edge
                    top = Edge(self.vertices[i][j], self.vertices[i - 1][j])
                    # put the edge into the graph
                    self.edges.append(top)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i - 1, j)] = top
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j)), relative2absolute((i-1, j))])
                if i - 1 >= 0 and j + 1 <= self.col - 1:
                    # create top right edge
                    topRight = Edge(self.vertices[i][j], self.vertices[i - 1][j + 1])
                    # put the edge into the graph
                    self.edges.append(topRight)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i - 1, j + 1)] = topRight
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j)), relative2absolute((i-1, j+1))])
                if j + 1 <= self.col - 1:
                    # create right edge
                    right = Edge(self.vertices[i][j], self.vertices[i][j + 1])
                    # put the edge into the graph
                    self.edges.append(right)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i, j + 1)] = right
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j)), relative2absolute((i, j+1))])
                if i + 1 <= self.row - 1 and j + 1 <= self.col - 1:
                    # create bottom right edge
                    bottomRight = Edge(self.vertices[i][j], self.vertices[i + 1][j + 1])
                    # put the edge into the graph
                    self.edges.append(bottomRight)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i + 1, j + 1)] = bottomRight
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j)), relative2absolute((i+1, j+1))])
                if i + 1 <= self.row - 1:
                    # create bottom edge
                    bottom = Edge(self.vertices[i][j], self.vertices[i + 1][j])
                    # put the edge into the graph
                    self.edges.append(bottom)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i + 1, j)] = bottom
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j)), relative2absolute((i+1, j))])
                if i + 1 <= self.row - 1 and j - 1 >= 0:
                    # create bottom left edge
                    bottomLeft = Edge(self.vertices[i][j], self.vertices[i + 1][j - 1])
                    # put the edge into the graph
                    self.edges.append(bottomLeft)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i + 1, j - 1)] = bottomLeft
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j)), relative2absolute((i+1, j-1))])
                if j - 1 >= 0:
                    # create left edge
                    left = Edge(self.vertices[i][j], self.vertices[i][j - 1])
                    # put the edge into the graph
                    self.edges.append(left)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i, j - 1)] = left
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j)), relative2absolute((i, j-1))])
        # initialize beta
        self.beta = 2 * sum(sigmas) / len(sigmas)
        # initialize weights of n-link edges
        for edge in self.edges:
            edge.calculateWeight(self.beta, self.square_distance_matrix)

    def _initialize_tlinks(self):
        # initialize K
        self.K = 1 + max([self.vertices[i][j].return_degree() for i in range(self.row) for j in range(self.col)])
        # initialize t-link edges
        for i in range(self.row):
            for j in range(self.col):
                # initialize source link
                edge_source = Edge(self.source, self.vertices[i][j])
                # initialize sink link
                edge_sink = Edge(self.vertices[i][j], self.sink)
                # calculate weights
                edge_source.calculateWeight(self.beta, self.square_distance_matrix, self.fore_seed_index, self.back_indices, self.K)
                edge_sink.calculateWeight(self.beta,  self.square_distance_matrix, self.fore_seed_index, self.back_indices, self.K)
                # connect edges with corresponding vertices
                self.source.edges[(i, j)] = edge_source
                self.vertices[i][j].edges["sink"] = edge_sink

    def _max_flow(self):
        """
        max flow algorithm
        :return:
        """

        def tree_cap(p, q):
            """
            return the capacity of the link from p to q if p belongs to S else from q to p
            :return:
            """
            if p.tree == 'S':
                # link from p to q
                return p.edges["sink"].capacity if q.isTerminal else p.edges[(q.row, q.col)].capacity
            elif p.tree == 'T':
                # link from q to p
                return q.edges["sink"].capacity if p.isTerminal else q.edges[(p.row, p.col)].capacity

        def collect_path(p):
            """
            collect path from source to p or from p to sink
            :return:
            """
            current_node = p
            path = []  # collected path
            while True:
                # push current node in the path
                path.append(current_node)
                # terminate if reach source or sink
                if current_node.isTerminal:
                    if current_node.tree == 'S':
                        path.reverse()
                    break
                else:
                    # obtain parent of p
                    parent = current_node.parent
                    # update current node
                    current_node = parent
            return path

        def findAugmentingPath(activeNodes):
            """
            find a augmenting path
            :return:
            """
            path = []  # augmenting path found
            while len(activeNodes) != 0:
                # pick an active node p
                p = activeNodes[0]
                # traverse all p's neighbors
                for coordinate, edge in p.edges.items():
                    # obtain neighbor
                    neighbor = edge.to
                    # if the edge is not saturated
                    if tree_cap(p, neighbor) > 0:
                        if neighbor.tree is None:
                            # push the neighbor into p's tree
                            neighbor.tree = p.tree
                            neighbor.parent = p
                            # set the neighbor active
                            if not neighbor.isActive:
                                activeNodes.append(neighbor)
                                neighbor.isActive = True
                        if p.tree != neighbor.tree:
                            # gather path
                            if p.tree == 'S':
                                path = collect_path(p)
                                path.extend(collect_path(neighbor))
                            else:
                                path = collect_path(neighbor)
                                path.extend(collect_path(p))
                            return path
                # set p passive
                activeNodes.pop(0)
                p.isActive = False
            return path

        def augmentation(path, orphans):
            # obtain edges capacity on the path
            edges_capacity = []
            for i in range(len(path) - 1):
                # obtain two adjacent nodes
                current_node = path[i]
                next_node = path[i + 1]
                # collect edges capacity
                capacity = current_node.edges["sink"].capacity if next_node.isTerminal else current_node.edges[
                    (next_node.row, next_node.col)].capacity
                edges_capacity.append(capacity)
            # find the bottleneck of the path
            min_flow = min(edges_capacity)
            # update residual graph
            for i in range(len(path) - 1):
                # obtain two adjacent nodes on the path
                p = path[i]
                q = path[i + 1]
                # obtain edge from p to q
                edge_pq = p.edges["sink"] if q.isTerminal else p.edges[(q.row, q.col)]
                # update p->q capacity
                edge_pq.capacity -= min_flow
                # update q->p capacity
                if not p.isTerminal and not q.isTerminal:
                    # obtain edge from q to p
                    edge_qp = q.edges[(p.row, p.col)]  # q->p
                    # update q->p capacity
                    edge_qp.capacity += min_flow
            # detect orphans
            for i in range(len(path) - 1):
                # obtain two adjacent nodes on the path
                p = path[i]
                q = path[i + 1]
                # obtain edge from p to q
                edge_pq = p.edges["sink"] if q.isTerminal else p.edges[(q.row, q.col)]
                # find orphan
                if edge_pq.capacity == 0:
                    # when p and q come from S
                    if p.tree == q.tree == 'S':
                        # make q an orphan
                        q.parent = None
                        orphans.append(q)
                    # when p and q come from T
                    elif p.tree == q.tree == 'T':
                        # make p an orphan
                        p.parent = None
                        orphans.append(p)

        def isOriginOrphan(node):
            """
            judge if the origin of a node is an orphan
            :return: true if an orphan
            """
            # initialize current node
            current_node = node
            while True:
                parent = current_node.parent
                # when reach the root
                if parent is None:
                    if current_node.isTerminal:
                        return False
                    else:
                        return True
                # update current node
                current_node = parent

        def adoption(orphans, activeNodes):
            while len(orphans) != 0:
                # pop an orphan
                orphan = orphans.pop()
                # traverse all the neighbors of the orphan
                for coordinate, edge in orphan.edges.items():
                    # obtain neighbor
                    neighbor = edge.to
                    # judge if the neighbor is a valid new parent
                    if neighbor.tree == orphan.tree and tree_cap(neighbor, orphan) > 0 and not isOriginOrphan(neighbor):
                        # if the neighbor is valid, then set the neighbor as new parent
                        orphan.parent = neighbor
                        break
                # if there is no valid parent for the orphan
                if orphan.parent is None:
                    # traverse all the neighbors of the orphan
                    for coordinate, edge in orphan.edges.items():
                        # obtain neighbor
                        neighbor = edge.to
                        if neighbor.tree == orphan.tree:
                            # make all the neighbor connected to the orphan active
                            if tree_cap(neighbor, orphan) > 0:
                                if not neighbor.isActive:
                                    neighbor.isActive = True
                                    activeNodes.append(neighbor)
                            # make the neighbor whose parent is current orphan as an orphan
                            if neighbor.parent == orphan:
                                neighbor.parent = None
                                orphans.append(neighbor)
                    orphan.tree = None
                    if orphan.isActive:
                        activeNodes.remove(orphan)
                        orphan.isActive = False

        # initialize tree sets
        A = [self.source, self.sink]  # set of active nodes
        # make source and sink active
        self.source.isActive = True
        self.sink.isActive = True
        O = []  # orphans

        # set tree of source and sink node
        self.source.tree = 'S'
        self.sink.tree = 'T'
        # conduct max flow algorithm
        while True:
            # print(f"\rSteps left: {len(A)}", end="")
            # find an augmenting path
            path = findAugmentingPath(A)
            # terminate if there is no more augmenting path found
            if len(path) == 0:
                break
            # augment on the path
            augmentation(path, O)
            # adopt orphans
            adoption(O, A)

    def segmentation(self):
        """
        conduct segmentation based on solving max flow problem
        :return:
        """
        foregrounds = []  # a list containing pixels classified in foreground
        backgrounds = []  # a list containing pixels classified in background
        # conduct max flow
        self._max_flow()
        # collect segmentation results
        for i in range(self.row):
            for j in range(self.col):
                # obtain a pixel
                vertex = self.vertices[i][j]
                # obtain the assignment
                if vertex.tree == 'S':
                    foregrounds.append((i, j))
                else:
                    backgrounds.append((i, j))
        return foregrounds, backgrounds


class SamplePoint:
    def __init__(self, x, y, mode, end=False):
        """
        a point on drawn curves
        :param x: the col coordinate
        :param y: the row coordinate
        :param mode: the mode of the curve that the point belongs to
        :param end: True if the point is the start point of a curve
        """
        self.y = y
        self.x = x
        self.mode = mode
        self.end = end


class GraphCutMaster:
    def __init__(self, image, distance_mode="Euclidean"):
        """
        integrate curve drawing and segmentation together
        fine_mode: "OPTICS" or "GraphBased" or None
        distance: "Euclidean" or "Mahalanobis"
        """
        self.image = image.copy()   # a copy of the image
        self.distance_mode = distance_mode
        # obtain features of the image
        self.features, self.qs, self.ks, self.vs = self._DINO_features(image)
        # obtain features of patches
        self.patch_features = self.ks[1:, :]
        # obtain the distance matrix and correlation matrix
        self.square_distance_matrix, self.correlation_matrix = self._square_distance_matrix(self.patch_features)
        # obtain the size of the image
        self.row = image.shape[0]
        self.col = image.shape[1]

    def _DINO_features(self, image):
        """
        return features of all patches at the last ViT block
        net: the model
        image: the BGR cv2 image, unnormalized
        """

        def _load_pretrained_DINO():
            import GraphCut.utils as utils
            import GraphCut.vision_transformer as vits
            # load the model
            net = vits.vit_small(patch_size=8, num_classes=0)
            # load the model's pre-trained weight
            net.cuda()
            utils.load_pretrained_weights(net, "./dino_deitsmall8_pretrain.pth", checkpoint_key=None,
                                          model_name="vit_small",
                                          patch_size=8)
            net.eval()
            return net

        def _preprocess(image):
            """
            preprocess the cv2 image
            """
            # transfer the image from bgr to rgb
            RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # resize the image to 224x224
            RGB_image = cv2.resize(RGB_image, (224, 224))
            # pytorch preprocessing
            transform = pth_transforms.Compose([
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            torch_image = transform(RGB_image)
            return torch_image.unsqueeze(0)  # 1, 3, 224, 224

        # preprocess the image
        torch_image = _preprocess(image)
        torch_image = torch_image.cuda()
        # load the model
        net = _load_pretrained_DINO()
        # obtain the output embeddings of DINO
        with torch.no_grad():
            features, qs, ks, vs = net.get_last_qkv(torch_image)
        if self.distance_mode == "Euclidean":
            return F.normalize(features.squeeze(0), p=2).detach().cpu().numpy(), \
                F.normalize(qs.permute(0, 2, 1, 3).reshape(1, 785, -1).squeeze(0), p=2).detach().cpu().numpy(), \
                F.normalize(ks.permute(0, 2, 1, 3).reshape(1, 785, -1).squeeze(0), p=2).detach().cpu().numpy(), \
                F.normalize(vs.permute(0, 2, 1, 3).reshape(1, 785, -1).squeeze(0), p=2).detach().cpu().numpy()
        elif self.distance_mode == "Mahalanobis":
            return features.squeeze(0).detach().cpu().numpy(), \
                qs.permute(0, 2, 1, 3).reshape(1, 785, -1).squeeze(0).detach().cpu().numpy(), \
                ks.permute(0, 2, 1, 3).reshape(1, 785, -1).squeeze(0).detach().cpu().numpy(), \
                vs.permute(0, 2, 1, 3).reshape(1, 785, -1).squeeze(0).detach().cpu().numpy()

    def _square_distance_matrix(self, features):
        """
        calculate squared distance matrix of each point
        params:
            features: 28x28x384
        """
        # resize features to -1, 384
        flatten_features = features.reshape(-1, 384)
        # obtain correlation
        correlation = flatten_features @ flatten_features.T
        if self.distance_mode == "Euclidean":
            # obtain length square of each feature
            features_square = np.sum(flatten_features ** 2, axis=1)
            # expand the feature squares into a square matrix by repeating
            features_square = np.repeat(features_square, 28 * 28, axis=0).reshape(-1, 28 * 28)
            # calculate the distance matrix
            square_distance_matrix = features_square - 2 * correlation + features_square.T
            np.fill_diagonal(square_distance_matrix, 0)
            square_distance_matrix[square_distance_matrix < 0] = 0
            return square_distance_matrix, correlation
        elif self.distance_mode == "Mahalanobis":
            # obtain x - y between each points
            differences = (flatten_features.reshape(784, 1, 384) - flatten_features.reshape(1, 784, 384)).reshape(-1, 384)
            # calculate covariance of features
            COV = np.cov(flatten_features, rowvar=False)
            # calculated squared Mahalanobis distance
            squared_Mahalanobis_matrix = np.sum((differences @ np.linalg.inv(COV)) * differences, axis=1).reshape(784, 784)
            return squared_Mahalanobis_matrix, correlation

    @staticmethod
    def _relative2absolute(indices):
        """
        transfer indices of matrix to array
        indices: np.array([[i,j],...])
        """
        return indices[:, 0] * 28 + indices[:, 1]

    @staticmethod
    def _absolute2relative(indices):
        """
        transfer indices of array to matrix
        indices: np.array([i,....])
        """
        j_s = indices % 28
        i_s = (indices - j_s) // 28
        return np.stack((i_s, j_s)).T

    def _background_indices(self, fore_seed, threshold=0):
        """
        obtain the indices of background features by thresholding on correlation map
        fore_seed: index of the foreground seed
        """
        return np.where(self.correlation_matrix[fore_seed, :]<threshold)[0]

    def _foreground_indices(self, fore_seed, threshold=0):
        """
        obtain the indices of foreground features by thresholding on correlation map
        fore_seed: index of the foreground seed
        """
        return np.where(self.correlation_matrix[fore_seed, :] > threshold)[0]

    def segmentation(self):
        """
        do segmentation: direct the user to draw curves and show the segmentation results
        :return:
        """
        # initialize graph cut
        graph = None
        foregrounds = []
        # traverse every feature point
        for i in tqdm(range(28*28), desc="traversing the image"):
            # make the current feature point as the foreground seed
            fore_seed_index = i
            # obtain guessed indices of foreground and background points
            fore_indices = self._foreground_indices(fore_seed_index)
            back_indices = self._background_indices(fore_seed_index)
            # do segmentation
            if graph is None:
                graph = GraphCut(self.square_distance_matrix, fore_seed_index, back_indices)
            else:
                graph.refresh_graph(fore_seed_index, back_indices)
            foreground, background = graph.segmentation()
            # vis
            foreground = np.array(foreground)
            background = np.array(background)
            canvas = np.zeros((28, 28, 3))
            canvas = canvas.astype(np.uint8)
            canvas[foreground[:, 0], foreground[:, 1]] = (255, 0, 0)
            canvas[background[:, 0], background[:, 1]] = (0, 0, 0)
            i_s, j_s = absolute2relative(i)
            canvas[i_s, j_s] = (0, 0, 255)
            canvas = cv2.resize(canvas, (self.col, self.row), interpolation=cv2.INTER_NEAREST)
            image = cv2.addWeighted(self.image, 0.6, canvas, 0.4, 0)
            cv2.imshow("image", image)
            cv2.waitKey(0)



if __name__ == '__main__':
    image_paths = ["/home/wzl/TokenCut/DINO/CMS_livingroom.PNG"] + [
        "/home/wzl/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/" + filename for filename in
        os.listdir("/home/wzl/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/")]
    for image_path in image_paths[8:]:
        # load a test image
        image = cv2.imread(image_path)
        # image = cv2.imread("/home/wzl/TokenCut/DINO/CMS_livingroom.PNG")
        # image = cv2.imread("./berlin_000011_000019_leftImg8bit.png")
        # image = cv2.resize(image, (int(0.5 * image.shape[1]), int(0.5 * image.shape[0])), cv2.INTER_AREA)
        graph_cut = GraphCutMaster(image, distance_mode="Mahalanobis")
        graph_cut.segmentation()
    cv2.destroyAllWindows()