import torch
from torchvision import transforms as pth_transforms
import math
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import os
import sys
import random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CRF'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crf import densecrf
from sklearn.cluster import MeanShift, estimate_bandwidth
from NCut import ncut, detect_box
import torch.nn.functional as F
import configs


def relative2absolute(index, col=28):
    """
    transfer indices of matrix to array
    index: (i, j)
    col: number of patches per row
    """
    return index[0] * col + index[1]


def absolute2relative(index, col=28):
    """
    transfer indices of array to matrix
    indices: np.array([i,....])
    """
    j_s = index % col
    i_s = (index - j_s) // col
    return i_s, j_s


def IoU(mask1, mask2):
    """
    calculate iou between mask1 and mask2
    mask1: (w, h) ndarray
    mask2: (w, h) ndarray
    """
    intersection = mask1 * mask2
    union = mask1 + mask2 - intersection
    return np.sum(intersection) / np.sum(union)


def Precision(mask, gt_mask):
    """
    calculate precision between mask1 and mask2
    mask: (w, h) ndarray
    gt_mask: (w, h) ndarray
    """
    intersection = mask * gt_mask
    return np.sum(intersection) / np.sum(mask)


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

    def calculateWeight(self, beta, square_distance_matrix, feat_col, fore_seed_index=None, back_indices=None, K=None):
        """
        calculate the weight of the edge
        :param beta: weight for n-link cost
        :param square_distance_matrix: squared distance matrix for each feature point
        :param feat_col: the number of patches per row on the feature map
        :param fore_seed_index: absolute index of the foreground seed
        :param back_indices: absolute indices of the background points
        :param K: weight for t-link ground truth
        :return:
        """
        # calculate weight for n-link
        if self._from.isTerminal is not True and self.to.isTerminal is not True:

            self.weight = 1 * math.exp(-square_distance_matrix[relative2absolute((self._from.row, self._from.col), col=feat_col), \
                relative2absolute((self.to.row, self.to.col), col=feat_col)] / beta) / \
                          math.sqrt((self._from.row - self.to.row) ** 2 + (self._from.col - self.to.col) ** 2)
        # calculate weight for t-link
        elif self._from.isTerminal:
            # obtain the absolute index of the node
            abs_index = relative2absolute((self.to.row, self.to.col), col=feat_col)
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
            abs_index = relative2absolute((self._from.row, self._from.col), col=feat_col)
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
                self.weight = - math.log(
                    np.exp(min_back_distance) / (np.exp(fore_distance) + np.exp(min_back_distance)))
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

    def __init__(self, square_distance_matrix, fore_seed_index, back_indices, row, col):
        """
        :param square_distance_matrix: squared distance matrix for each feature point
        :param fore_seed_index: absolute index of the foreground seed
        :param back_indices: absolute indices ot the background
        :param row, col: size of the feature map
        :return:
        """
        self.square_distance_matrix = square_distance_matrix
        self.fore_seed_index = fore_seed_index
        self.back_indices = back_indices
        # obtain the size of the feature map
        self.row = row
        self.col = col
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
                    sigmas.append(
                        self.square_distance_matrix[relative2absolute((i, j), self.col), relative2absolute((i - 1, j - 1), self.col)])
                if i - 1 >= 0:
                    # create top edge
                    top = Edge(self.vertices[i][j], self.vertices[i - 1][j])
                    # put the edge into the graph
                    self.edges.append(top)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i - 1, j)] = top
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j), self.col), relative2absolute((i - 1, j), self.col)])
                if i - 1 >= 0 and j + 1 <= self.col - 1:
                    # create top right edge
                    topRight = Edge(self.vertices[i][j], self.vertices[i - 1][j + 1])
                    # put the edge into the graph
                    self.edges.append(topRight)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i - 1, j + 1)] = topRight
                    # save camera noise
                    sigmas.append(
                        self.square_distance_matrix[relative2absolute((i, j), self.col), relative2absolute((i - 1, j + 1), self.col)])
                if j + 1 <= self.col - 1:
                    # create right edge
                    right = Edge(self.vertices[i][j], self.vertices[i][j + 1])
                    # put the edge into the graph
                    self.edges.append(right)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i, j + 1)] = right
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j), self.col), relative2absolute((i, j + 1), self.col)])
                if i + 1 <= self.row - 1 and j + 1 <= self.col - 1:
                    # create bottom right edge
                    bottomRight = Edge(self.vertices[i][j], self.vertices[i + 1][j + 1])
                    # put the edge into the graph
                    self.edges.append(bottomRight)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i + 1, j + 1)] = bottomRight
                    # save camera noise
                    sigmas.append(
                        self.square_distance_matrix[relative2absolute((i, j), self.col), relative2absolute((i + 1, j + 1), self.col)])
                if i + 1 <= self.row - 1:
                    # create bottom edge
                    bottom = Edge(self.vertices[i][j], self.vertices[i + 1][j])
                    # put the edge into the graph
                    self.edges.append(bottom)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i + 1, j)] = bottom
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j), self.col), relative2absolute((i + 1, j), self.col)])
                if i + 1 <= self.row - 1 and j - 1 >= 0:
                    # create bottom left edge
                    bottomLeft = Edge(self.vertices[i][j], self.vertices[i + 1][j - 1])
                    # put the edge into the graph
                    self.edges.append(bottomLeft)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i + 1, j - 1)] = bottomLeft
                    # save camera noise
                    sigmas.append(
                        self.square_distance_matrix[relative2absolute((i, j), self.col), relative2absolute((i + 1, j - 1), self.col)])
                if j - 1 >= 0:
                    # create left edge
                    left = Edge(self.vertices[i][j], self.vertices[i][j - 1])
                    # put the edge into the graph
                    self.edges.append(left)
                    # connect the edge with corresponding vertex
                    self.vertices[i][j].edges[(i, j - 1)] = left
                    # save camera noise
                    sigmas.append(self.square_distance_matrix[relative2absolute((i, j), self.col), relative2absolute((i, j - 1), self.col)])
        # initialize beta
        self.beta = 2 * sum(sigmas) / len(sigmas)
        # initialize weights of n-link edges
        for edge in self.edges:
            edge.calculateWeight(self.beta, self.square_distance_matrix, feat_col=self.col)

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
                edge_source.calculateWeight(self.beta, self.square_distance_matrix, self.col, self.fore_seed_index,
                                            self.back_indices, self.K)
                edge_sink.calculateWeight(self.beta, self.square_distance_matrix, self.col, self.fore_seed_index,
                                          self.back_indices, self.K)
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
    def __init__(self, image, scan_distance_mode="Euclidean", tau=0.2, eps=1e-5, patch_size=8, resize_mode="fixed"):
        """
        UnionCut core functions
        scan_distance_mode: "Euclidean" or "Mahalanobis" for scanning
        tau: threshold for attention map construction
        eps: graph edge weight of non-relative links
        patch_size: size of the patch for UnionCut
        resize_mode: the input size of the image for UnionCut, fixed -> 224x224; flexible - > ratio does not change
        """
        self.image = image.copy()  # a copy of the image
        # save parameters for NCut
        self.tau = tau
        self.eps = eps
        # obtain the size of the image
        self.row = image.shape[0]
        self.col = image.shape[1]
        # initialize the maximum exploration times per image
        self.N = 1
        # initialize the patch size
        self.patch_size = patch_size
        # initialize preprocess mode of resize
        self.resize_mode = resize_mode
        # obtain the size of the feature map
        if self.resize_mode == "fixed":
            self.feat_row, self.feat_col = 224 // patch_size, 224 // patch_size
        else:
            self.feat_row, self.feat_col = int(np.ceil(self.row / self.patch_size)), int(np.ceil(self.col / self.patch_size))
        # obtain features of the image
        self.features, self.qs, self.ks, self.vs = self._DINO_features(image, patch_size, resize_mode, self.feat_row, self.feat_col)
        # obtain the distance matrix and correlation matrix
        self.square_distance_matrix, self.correlation_matrix = self._square_distance_matrix(
            self._norm_features(self.ks[1:, :]), distance_mode=scan_distance_mode, feat_row=self.feat_row, feat_col=self.feat_col)

    @staticmethod
    def _DINO_features(image, patch_size, resize_mode, feat_row, feat_col):
        """
        return features of all patches at the last ViT block
        net: the model
        image: the BGR cv2 image, unnormalized
        """

        def _load_pretrained_DINO():
            import utils as utils
            import vision_transformer as vits
            # load the model
            if patch_size == 8:
                net = vits.vit_small(patch_size=8, num_classes=0)
                # load the model's pre-trained weight
                # net.cuda()
                utils.load_pretrained_weights(net,
                                              configs.DINO_vit_small_8,
                                              checkpoint_key=None,
                                              model_name="vit_small",
                                              patch_size=8)
            elif patch_size == 16:
                net = vits.vit_small(patch_size=16, num_classes=0)
                # load the model's pre-trained weight
                # net.cuda()
                utils.load_pretrained_weights(net,
                                              configs.DINO_vit_small_16,
                                              checkpoint_key=None,
                                              model_name="vit_small",
                                              patch_size=16)
            net.eval()
            return net

        def _preprocess(image, mode="fixed"):
            """
            preprocess the cv2 image
            mode: fixed -> 224 * 224; flexible -> ratio does not change
            """
            # transfer the image from bgr to rgb
            RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if mode == "fixed":
                # resize the image to 224x224
                RGB_image = cv2.resize(RGB_image, (224, 224))
            elif mode == "flexible":
                zeros_array = np.zeros((feat_row * patch_size, feat_col * patch_size, 3)).astype(np.uint8)
                zeros_array[:RGB_image.shape[0], :RGB_image.shape[1], :] = RGB_image.copy()
                RGB_image = zeros_array
            # pytorch preprocessing
            transform = pth_transforms.Compose([
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            torch_image = transform(RGB_image)
            return torch_image.unsqueeze(0)  # 1, 3, 224, 224

        # preprocess the image
        torch_image = _preprocess(image, mode=resize_mode)
        # torch_image = torch_image.cuda()
        # load the model
        net = _load_pretrained_DINO()
        # obtain the output embeddings of DINO
        with torch.no_grad():
            features, qs, ks, vs = net.get_last_qkv(torch_image)
        return features.squeeze(0).detach().cpu(), \
            qs.permute(0, 2, 1, 3).reshape(1, feat_row * feat_col + 1, -1).squeeze(0).detach().cpu(), \
            ks.permute(0, 2, 1, 3).reshape(1, feat_row * feat_col + 1, -1).squeeze(0).detach().cpu(), \
            vs.permute(0, 2, 1, 3).reshape(1, feat_row * feat_col + 1, -1).squeeze(0).detach().cpu()

    @staticmethod
    def _norm_features(features):
        """
        normalize features
        """
        return F.normalize(features, p=2).numpy()

    @staticmethod
    def _square_distance_matrix(features, distance_mode, feat_row, feat_col):
        """
        calculate squared distance matrix of each point
        params:
            features: (self.feat_row * self.feat_col, 384)
        """
        # resize features to -1, 384
        flatten_features = features.reshape(-1, 384)
        # obtain correlation
        correlation = flatten_features @ flatten_features.T
        if distance_mode == "Euclidean":
            # obtain length square of each feature
            features_square = np.sum(flatten_features ** 2, axis=1)
            # expand the feature squares into a square matrix by repeating
            features_square = np.repeat(features_square, feat_row * feat_col, axis=0).reshape(-1, feat_row * feat_col)
            # calculate the distance matrix
            square_distance_matrix = features_square - 2 * correlation + features_square.T
            np.fill_diagonal(square_distance_matrix, 0)
            square_distance_matrix[square_distance_matrix < 0] = 0
            return square_distance_matrix, correlation
        elif distance_mode == "Mahalanobis":
            # obtain x - y between each points
            differences = (flatten_features.reshape(feat_row * feat_col, 1, 384) - flatten_features.reshape(1, feat_row * feat_col, 384)).reshape(-1,
                                                                                                                  384)
            # calculate covariance of features
            COV = np.cov(flatten_features, rowvar=False)
            # calculated squared Mahalanobis distance
            squared_Mahalanobis_matrix = np.sum((differences @ np.linalg.inv(COV)) * differences, axis=1).reshape(feat_row * feat_col,
                                                                                                                  feat_row * feat_col)
            return squared_Mahalanobis_matrix, correlation

    @staticmethod
    def _relative2absolute(indices, feat_col):
        """
        transfer indices of matrix to array
        indices: np.array([[i,j],...])
        """
        return indices[:, 0] * feat_col + indices[:, 1]

    @staticmethod
    def _absolute2relative(indices, feat_col):
        """
        transfer indices of array to matrix
        indices: np.array([i,....])
        """
        j_s = indices % feat_col
        i_s = (indices - j_s) // feat_col
        return np.stack((i_s, j_s)).T

    def _background_indices(self, fore_seed, threshold=0):
        """
        obtain the indices of background features by thresholding on correlation map
        fore_seed: index of the foreground seed
        """
        return np.where(self.correlation_matrix[fore_seed, :] < threshold)[0]

    def _foreground_indices(self, fore_seed, threshold=0):
        """
        obtain the indices of foreground features by thresholding on correlation map
        fore_seed: index of the foreground seed
        """
        return np.where(self.correlation_matrix[fore_seed, :] > threshold)[0]

    def _disconnection_detection(self, mask):
        """
        split disconnected masks in the given mask
        """
        #initialize a list for saving all isolated masks
        isolated_masks = []
        # resize the mask
        mask = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # connectivity detection
        num_labels, labels = cv2.connectedComponents(mask)
        for j in range(1, num_labels):
            # obtain an isolate mask
            isolate_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
            isolate_mask[labels == j] = 1
            # apply crf
            isolate_mask = densecrf(self.image, isolate_mask)
            # collect the mask
            if np.sum(isolate_mask) != 0:
                isolated_masks.append(isolate_mask)
        return isolated_masks

    @staticmethod
    def _check_num_fg_corners(mask):
        # check number of corners belonging to the foreground
        top_l, top_r, bottom_l, bottom_r = mask[0][0], mask[0][-1], mask[-1][0], mask[-1][
            -1]
        nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
        return nc

    @staticmethod
    def _visualization(masks):
        """
        visualize each component with different color
        :return:
        """
        import random
        # obtain the number of colours
        colour_num = masks.shape[0]
        # generate random colours along each channel
        R = random.sample(list(range(256)), colour_num)
        G = random.sample(list(range(256)), colour_num)
        B = random.sample(list(range(256)), colour_num)
        # initialize an empty tensor to represent the visualization map
        visual_map = np.zeros((masks.shape[1], masks.shape[2], 3))
        # colour each mask
        for i in range(colour_num):
            visual_map[masks[i, :, :] == 1] = [R[i], G[i], B[i]]
        # transfer the map into uint8
        visual_map = visual_map.astype(np.uint8)
        return visual_map

    def segmentation(self, integrity=True, small_thresh=0.01, vis=True):
        """
        TokenCut + UnionCut
        integrity: True to avoid finner segmentation
        fine_thresh: thresh for judging good finner segmentation
        small_thresh: thresh for removing too small masks
        :return:
        """
        # initialize graph cut
        graph = None
        foregrounds = []
        # traverse every feature point
        for i in tqdm(range(self.feat_row * self.feat_col), desc="traversing the image"):
            # make the current feature point as the foreground seed
            fore_seed_index = i
            # obtain guessed indices of foreground and background points
            back_indices = self._background_indices(fore_seed_index)
            # do segmentation
            if graph is None:
                graph = GraphCut(self.square_distance_matrix, fore_seed_index, back_indices, self.feat_row, self.feat_col)
            else:
                graph.refresh_graph(fore_seed_index, back_indices)
            foreground, background = graph.segmentation()
            foreground = np.array(foreground)
            # obtain mask of the foreground
            mask = np.zeros((self.feat_row, self.feat_col))
            mask[foreground[:, 0], foreground[:, 1]] = 1
            foregrounds.append(mask)

        # vote for background area
        accumulated_attention = np.sum(np.stack(foregrounds), axis=0)
        # normalize the attention
        norm_accumulated_attention = 255 - (255 * (accumulated_attention - np.min(accumulated_attention)) / (
                    np.max(accumulated_attention) - np.min(accumulated_attention))).astype(np.uint8)
        # cluster the attention map with mean shift
        flatten_attention = norm_accumulated_attention.reshape(-1, 1)
        bandwidth = estimate_bandwidth(flatten_attention)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(flatten_attention)
        # obtain the cluster id of the minimum cluster center
        background_ids = np.argsort(ms.cluster_centers_.reshape(-1))
        background_id = background_ids[len(background_ids) // 2 - 1]
        # remove background mask
        sub_members = ms.labels_ == background_id
        max_thresh = np.max(flatten_attention[sub_members].reshape(-1))
        indices = (norm_accumulated_attention <= max_thresh).astype(np.uint8)
        fore_combined_mask = np.zeros((self.feat_row, self.feat_col), dtype=np.uint8)
        fore_combined_mask[indices == 0] = 1
        fore_combined_mask[indices == 1] = 0
        # reverse the mask if current foreground covering 4 corners
        if self._check_num_fg_corners(cv2.erode(densecrf(self.image, fore_combined_mask), kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))) >= 4 or self._check_num_fg_corners(
                fore_combined_mask) >= 4:
            indices = 1 - indices
            fore_combined_mask[indices == 0] = 1
            fore_combined_mask[indices == 1] = 0
        # resize the mask to the NCut output scale
        NCut_row, NCut_col = int(np.ceil(self.row / 16)), int(np.ceil(self.col / 16))
        fore_combined_mask = cv2.resize(fore_combined_mask, (NCut_col, NCut_row), cv2.INTER_NEAREST)
        # hierarchical NCut
        # obtain features of the image for NCut
        _, _, ks, _ = self._DINO_features(self.image, patch_size=16, resize_mode="flexible", feat_row=NCut_row, feat_col=NCut_col)
        _, cos_similarity_matrix = self._square_distance_matrix(self._norm_features(ks[1:, :]), distance_mode="Euclidean", feat_row=NCut_row, feat_col=NCut_col)
        # initialize results list
        masks = []
        for i in range(self.N):
            # NCut
            try:
                bipartition, seed, second_smallest_vec = ncut(cos_similarity_matrix, self.tau, self.eps, feat_row=NCut_row, feat_col=NCut_col)
                # obtain the seed for bipartition
                seed_neg = np.argmin(second_smallest_vec)
                seed_pos = np.argmax(second_smallest_vec)
                if bipartition[seed] != 1:
                    # reverse the mask
                    bipartition = ~bipartition
                    # change the seed
                    seed = seed_neg if bipartition[seed_neg] else seed_pos
                # extract the principal component
                bipartition = bipartition.reshape([NCut_row, NCut_col]).astype(float)
                # predict BBox
                pred, _, objects, cc = detect_box(bipartition, seed,
                                                           [NCut_row, NCut_col],
                                                           scales=[16, 16],
                                                           initial_im_size=[
                                                               NCut_row * 16,
                                                               NCut_col * 16])
                mask = np.zeros([NCut_row, NCut_col])
                mask[cc[0], cc[1]] = 1
                bipartition = (mask == 1).reshape(-1)
            except:
                break
            # obtain the abs indices the subsub-region
            sub_fore_indices, sub_back_indices = np.arange(NCut_row * NCut_col)[bipartition], np.arange(NCut_row * NCut_col)[~bipartition]
            # transfer abs indices to relative indices
            sub_fore_relative_indices = self._absolute2relative(sub_fore_indices, feat_col=NCut_col)
            sub_back_relative_indices = self._absolute2relative(sub_back_indices, feat_col=NCut_col)
            # obtain mask
            sub_fore_mask, sub_back_mask = np.zeros((NCut_row, NCut_col), dtype=np.uint8), np.zeros((NCut_row, NCut_col), dtype=np.uint8)
            sub_fore_mask[sub_fore_relative_indices[:, 0], sub_fore_relative_indices[:, 1]] = 1
            sub_back_mask[sub_back_relative_indices[:, 0], sub_back_relative_indices[:, 1]] = 1
            # # judge if the mask is the fore mask
            # if Precision(sub_fore_mask, fore_combined_mask) < 0.5:
            #     # fore looks bad
            #     if self._check_num_fg_corners(sub_back_mask) < 4:
            #         # and back is good at corners, switch
            #         sub_fore_mask, sub_back_mask = sub_back_mask, sub_fore_mask
            #         sub_fore_indices, sub_back_indices = sub_back_indices, sub_fore_indices
            # else:
            #     # fore looks good
            #     if self._check_num_fg_corners(sub_fore_mask) >= 4:
            #         # but fore is bad at corners,  switch
            #         sub_fore_mask, sub_back_mask = sub_back_mask, sub_fore_mask
            #         sub_fore_indices, sub_back_indices = sub_back_indices, sub_fore_indices
            # stop if the IOU between the mask and the previous one is large or it is not covered by the combined fore mask

            # judge if the mask is the fore mask
            if len(masks) == 0:
                # this is the first mask
                if self._check_num_fg_corners(cv2.erode(densecrf(self.image, sub_fore_mask), kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))) >= 3:
                    sub_fore_mask, sub_back_mask = sub_back_mask, sub_fore_mask
                    sub_fore_indices, sub_back_indices = sub_back_indices, sub_fore_indices
                elif Precision(sub_fore_mask, fore_combined_mask) < Precision(sub_back_mask, fore_combined_mask):
                    sub_fore_mask, sub_back_mask = sub_back_mask, sub_fore_mask
                    sub_fore_indices, sub_back_indices = sub_back_indices, sub_fore_indices
                    #self._check_num_fg_corners(sub_fore_mask) >= 3 or
                    if self._check_num_fg_corners(
                            cv2.erode(densecrf(self.image, sub_fore_mask),
                                      kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))) >= 3:
                        sub_fore_mask, sub_back_mask = sub_back_mask, sub_fore_mask
                        sub_fore_indices, sub_back_indices = sub_back_indices, sub_fore_indices
            else:
                if IoU(sub_fore_mask, masks[-1]) > 0.5 or Precision(sub_fore_mask, fore_combined_mask) < 0.75:
                    break
                # remove previous mask from the current mask
                for mask in masks:
                    if Precision(mask, sub_fore_mask) > 0.5:
                        sub_fore_mask = sub_fore_mask - mask
                        sub_fore_mask[sub_fore_mask < 0] = 0
            # save the mask
            masks.append(sub_fore_mask)
            # mask out the mask
            cos_similarity_matrix[sub_fore_indices, :] = self.eps
            cos_similarity_matrix[:, sub_fore_indices] = self.eps
            # judge if the bipartition should be finished
            # calculate the overall segmented mask
            overall_segmented_mask = np.sum(np.stack(masks), axis=0)
            overall_segmented_mask[overall_segmented_mask > 1] = 1
            if Precision(fore_combined_mask, overall_segmented_mask) >= 0.8:
                # most of the foreground has been segmented
                break

        # apply crf
        crf_masks = []
        for mask in masks:
            crf_masks.extend(self._disconnection_detection(mask))
        # sort the masks by area in descending order
        crf_masks.sort(key=lambda x: np.sum(x), reverse=True)
        masks = np.stack(crf_masks)
        # judge if the fine segmentation is robust
        if integrity:
            # obtain the total area of overall mask
            fore_total = np.sum(masks, axis=0)
            fore_total[fore_total > 1] = 1
            fore_total = np.sum(fore_total)
            # remove too small masks
            masks = masks[np.sum(masks, axis=(1, 2)) / fore_total > small_thresh]
            # make sure one mask is returned
            if len(masks) == 0:
                masks = np.stack(self._disconnection_detection(crf_masks[0]))
        # visualization
        if vis:
            # apply crf
            import matplotlib.pyplot as plt
            hist_frame = cv2.calcHist([norm_accumulated_attention], [0], None, [256], [0, 255])
            # plt.plot(hist_frame, color='b')
            # plt.grid(True)
            cv2.imshow('image', image)
            cv2.imshow("heat map", cv2.resize(norm_accumulated_attention, (self.col, self.row)))
            # cv2.imshow("fore total", 255 * densecrf(self.image, fore_combined_mask))
            cv2.imshow("fore union", cv2.resize(255 * fore_combined_mask, (image.shape[1], image.shape[0]),
                                                interpolation=cv2.INTER_NEAREST))
            # print(len(background_ids), max_thresh)
            # plt.show()
            colorful_masks = self._visualization(masks)
            cv2.imshow("masks", colorful_masks)

            # waiting for user interaction
            pressedKey = cv2.waitKey(0) & 0xFF
            if pressedKey == ord('s'):
                # save the figue and mask
                cv2.imwrite("./demo/" + image_path.split("/")[-1], image)
                cv2.imwrite("./demo/" + image_path.split("/")[-1].split(".")[0] + "_mask.jpg", colorful_masks)
                cv2.imwrite("./demo/" + image_path.split("/")[-1].split(".")[0] + "_union.jpg", 255 * fore_combined_mask)
            elif pressedKey == ord('q'):
                # press q to quit
                cv2.destroyAllWindows()
                sys.exit()
        return masks, fore_combined_mask

    def returnUnion(self, vis=True, crf=False):
        """
        return the foreground union detected by UnionCut
        :return:
        """
        # initialize graph cut
        graph = None
        foregrounds = []
        # traverse every feature point
        for i in tqdm(range(self.feat_row * self.feat_col), desc="traversing the image"):
            # make the current feature point as the foreground seed
            fore_seed_index = i
            # obtain guessed indices of foreground and background points
            back_indices = self._background_indices(fore_seed_index)
            # do segmentation
            if graph is None:
                graph = GraphCut(self.square_distance_matrix, fore_seed_index, back_indices, self.feat_row, self.feat_col)
            else:
                graph.refresh_graph(fore_seed_index, back_indices)
            foreground, background = graph.segmentation()
            foreground = np.array(foreground)
            # obtain mask of the foreground
            mask = np.zeros((self.feat_row, self.feat_col))
            mask[foreground[:, 0], foreground[:, 1]] = 1
            foregrounds.append(mask)

        # vote for background area
        accumulated_attention = np.sum(np.stack(foregrounds), axis=0)
        # normalize the attention
        norm_accumulated_attention = 255 - (255 * (accumulated_attention - np.min(accumulated_attention)) / (
                np.max(accumulated_attention) - np.min(accumulated_attention))).astype(np.uint8)
        # cluster the attention map with mean shift
        flatten_attention = norm_accumulated_attention.reshape(-1, 1)
        bandwidth = estimate_bandwidth(flatten_attention)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(flatten_attention)
        # obtain the cluster id of the minimum cluster center
        background_ids = np.argsort(ms.cluster_centers_.reshape(-1))
        background_id = background_ids[len(background_ids) // 2 - 1]
        # remove background mask
        sub_members = ms.labels_ == background_id
        max_thresh = np.max(flatten_attention[sub_members].reshape(-1))
        indices = (norm_accumulated_attention <= max_thresh).astype(np.uint8)
        fore_combined_mask = np.zeros((self.feat_row, self.feat_col), dtype=np.uint8)
        fore_combined_mask[indices == 0] = 1
        fore_combined_mask[indices == 1] = 0
        # reverse the mask if current foreground covering 4 corners
        if self._check_num_fg_corners(cv2.erode(densecrf(self.image, fore_combined_mask),
                                                kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))) >= 4 \
                or self._check_num_fg_corners(fore_combined_mask) >= 4:
            indices = 1 - indices
            fore_combined_mask[indices == 0] = 1
            fore_combined_mask[indices == 1] = 0
        # apply crf
        if crf:
            fore_combined_mask = densecrf(self.image, fore_combined_mask)
        if vis:
            cv2.imshow('image', image)
            cv2.imshow("fore union", cv2.resize(255 * fore_combined_mask, (image.shape[1], image.shape[0]),
                                                interpolation=cv2.INTER_NEAREST))
            cv2.imshow("crf union", densecrf(self.image, fore_combined_mask))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return fore_combined_mask


def TokenGraphCut(image, integrity=True, scan_distance_mode="Euclidean", small_thresh=0.01, vis=False, patch_size=16, resize_mode="fixed"):
    """
    implementation of TokenGraphCut
    image: image to be segmented
    integrity: True to avoid finner segmentation
    fine_mode: "OPTICS" or "GraphBased" or "NCut" or None
    scan_distance_mode: "Euclidean" or "Mahalanobis" for scanning
    cluster_distance_mode: "Euclidean" or "Mahalanobis" for finner segmentation
    fine_thresh: thresh for judging good finner segmentation
    small_thresh: thresh for removing too small masks
    """
    # initialize graph cut
    graph_cut = GraphCutMaster(image, scan_distance_mode=scan_distance_mode, tau=0.2, patch_size=patch_size, resize_mode=resize_mode)
    masks, fore_combined_mask = graph_cut.segmentation(integrity=integrity, small_thresh=small_thresh, vis=vis)
    return masks, fore_combined_mask


def UnionCut(image, scan_distance_mode="Euclidean", vis=False, crf=False):
    """
    implementation of UnionCut
    """
    # initialize graph cut
    graph_cut = GraphCutMaster(image, scan_distance_mode=scan_distance_mode, tau=0.15, patch_size=8, resize_mode="fixed")
    fore_union = graph_cut.returnUnion(vis=vis, crf=crf)
    return fore_union


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TokenCut+UnionCut demo script')
    # default arguments
    parser.add_argument('--img-folder', type=str, default=None, help='folder of images')
    args = parser.parse_args()

    image_paths = [os.path.join(args.img_folder, filename) for filename in os.listdir(args.img_folder)]

    for image_path in image_paths:
        # load a test image
        image = cv2.imread(image_path)
        TokenGraphCut(image, vis=True, patch_size=8, resize_mode="fixed")
