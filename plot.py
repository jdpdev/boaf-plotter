import datetime
import math
from itertools import product
from random import choices, randint
from typing import List, Tuple

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import shapely.affinity as affinity
import shapely.geometry as geo
import shapely.ops as ops
import vsketch
from sklearn.neighbors import NearestNeighbors

DISPLAY_RATIO = 1.5

def display(sketch):
    return sketch.display("matplotlib", fig_size=(6*DISPLAY_RATIO,4*DISPLAY_RATIO))

vsk = vsketch.Vsketch()
vsk.size("6x4in", center=True)

W = vsk.width
H = vsk.height

np.random.seed(24601)
points = np.random.random((10, 2))
points = np.multiply(points, [W, H])
vor = Voronoi(points)
voronoi_plot_2d(vor)

lines = [
    geo.LineString(vor.vertices[line])
    for line in vor.ridge_vertices
    if -1 not in line
]

for point in points:
    vsk.fill(1)
    vsk.stroke(1)

index = 2
for poly in ops.polygonize(lines):
    #vsk.stroke(index + 1)
    vsk.fill(index)
    vsk.geometry(poly)
    index += 1

display(vsk)



"""
MARGIN = 25
RADIUS = 21
SPACING = RADIUS * (2 + 0.05)

xrange = np.arange(MARGIN, int(W - MARGIN), SPACING)
yrange = np.arange(MARGIN, int(H - MARGIN), SPACING)
points = []
for x, y in product(xrange, yrange):
    points.append((x, y))

def knn_grid(
    grid_points: List[Tuple[float, float]], n_centers: int = 3, metric: str = "euclidean"
):
    centers = choices(grid_points, k=n_centers)  # choose k points to be "centers"
    neigh = NearestNeighbors(n_neighbors=1, metric=metric)
    neigh.fit(centers)  # fit knn on those centers as the 'label'
    # for each point, return the 1st nearest neighbor
    closest_point = neigh.kneighbors(points, 1, return_distance=False)
    labels = list(c[0] for c in closest_point)
    return labels, centers

labels, centers = knn_grid(points)

for p, label in zip(points, labels):
    vsk.stroke(label + 1)
    vsk.fill(label+1)
    buffer = 15 if p in centers else 5
    vsk.geometry(geo.Point(*p).buffer(buffer))

display(vsk)
"""