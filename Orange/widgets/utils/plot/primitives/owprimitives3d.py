import os
import re
from plot.owplot3d import Symbol, enum
import numpy

GeometryType = enum('SOLID_3D', 'SOLID_2D', 'EDGE_3D', 'EDGE_2D')

def normalize(vec):
    return vec / numpy.sqrt(numpy.sum(vec**2))

def clamp(value, min, max):
    if value < min:
        return min
    if value > max:
        return max
    return value

def normal_from_points(p1, p2, p3):
    if isinstance(p1, (list, tuple)):
        v1 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]
        v2 = [p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]]
    else:
        v1 = p2 - p1
        v2 = p3 - p1
    return normalize(numpy.cross(v1, v2))

symbol_map = {
    Symbol.RECT:      'cube.obj',
    Symbol.TRIANGLE:  'pyramid.obj',
    Symbol.DTRIANGLE: 'dpyramid.obj',
    Symbol.CIRCLE:    'sphere.obj',
    Symbol.LTRIANGLE: 'lpyramid.obj',
    Symbol.DIAMOND:   'octahedron.obj',
    Symbol.WEDGE:     'wedge.obj',
    Symbol.LWEDGE:    'lwedge.obj',
    Symbol.CROSS:     'cross.obj',
    Symbol.XCROSS:    'xcross.obj'
}

symbol_map_2d = {
    Symbol.RECT:      'rect.obj',
    Symbol.TRIANGLE:  'triangle.obj',
    Symbol.DTRIANGLE: 'dtriangle.obj',
    Symbol.CIRCLE:    'circle.obj',
    Symbol.LTRIANGLE: 'ltriangle.obj',
    Symbol.DIAMOND:   'diamond.obj',
    Symbol.WEDGE:     'wedge_2d.obj',
    Symbol.LWEDGE:    'lwedge_2d.obj',
    Symbol.CROSS:     'cross_2d.obj',
    Symbol.XCROSS:    'xcross_2d.obj'
}

symbol_edge_map = {
    Symbol.RECT:      'cube_edges.obj',
    Symbol.TRIANGLE:  'pyramid_edges.obj',
    Symbol.DTRIANGLE: 'dpyramid_edges.obj',
    Symbol.CIRCLE:    'sphere_edges.obj',
    Symbol.LTRIANGLE: 'lpyramid_edges.obj',
    Symbol.DIAMOND:   'octahedron_edges.obj',
    Symbol.WEDGE:     'wedge_edges.obj',
    Symbol.LWEDGE:    'lwedge_edges.obj',
    Symbol.CROSS:     'cross_edges.obj',
    Symbol.XCROSS:    'xcross_edges.obj'
}

symbol_edge_map_2d = {
    Symbol.RECT:      'rect_edges.obj',
    Symbol.TRIANGLE:  'triangle_edges.obj',
    Symbol.DTRIANGLE: 'dtriangle_edges.obj',
    Symbol.CIRCLE:    'circle_edges.obj',
    Symbol.LTRIANGLE: 'ltriangle_edges.obj',
    Symbol.DIAMOND:   'diamond_edges.obj',
    Symbol.WEDGE:     'wedge_2d_edges.obj',
    Symbol.LWEDGE:    'lwedge_2d_edges.obj',
    Symbol.CROSS:     'cross_2d_edges.obj',
    Symbol.XCROSS:    'xcross_2d_edges.obj'
}

_symbol_data = {} # Cache: contains triangles + vertices normals for each needed symbol.
_symbol_data_2d = {}
_symbol_edges = {}
_symbol_edges_2d = {}

def parse_obj(file_name):
    if not os.path.exists(file_name):
        # Try to load the file from primitives/ folder if given only name (path missing).
        file_name = os.path.join(os.path.dirname(__file__), file_name)
    lines = open(file_name).readlines()
    normals_lines =  filter(lambda line: line.startswith('vn '), lines)
    vertices_lines = filter(lambda line: line.startswith('v '), lines)
    faces_lines =    filter(lambda line: line.startswith('f '), lines)
    normals =  [map(float, line.split()[1:]) for line in normals_lines]
    vertices = [map(float, line.split()[1:]) for line in vertices_lines]
    if len(normals) > 0:
        pattern = r'f (\d+)//(\d+) (\d+)//(\d+) (\d+)//(\d+)'
        faces = [map(int, re.match(pattern, line).groups()) for line in faces_lines]
        triangles = [[vertices[face[0]-1],
                      vertices[face[2]-1],
                      vertices[face[4]-1],
                      normals[face[1]-1],
                      normals[face[3]-1],
                      normals[face[5]-1]] for face in faces]
    else:
        faces = [map(int, line.split()[1:]) for line in faces_lines]
        triangles = []
        for face in faces:
            v0 = vertices[face[0]-1]
            v1 = vertices[face[1]-1]
            v2 = vertices[face[2]-1]
            normal = normal_from_points(v0, v1, v2)
            triangles.append([v0, v1, v2, normal, normal, normal])
    return triangles

def get_symbol_geometry(symbol, geometry_type):
    if not Symbol.is_valid(symbol):
        print('Invalid symbol!')
        return []

    if geometry_type == GeometryType.SOLID_3D:
        cache = _symbol_data
        name_map = symbol_map
    elif geometry_type == GeometryType.SOLID_2D:
        cache = _symbol_data_2d
        name_map = symbol_map_2d
    elif geometry_type == GeometryType.EDGE_3D:
        cache = _symbol_edges
        name_map = symbol_edge_map
    elif geometry_type == GeometryType.EDGE_2D:
        cache = _symbol_edges_2d
        name_map = symbol_edge_map_2d
    else:
        print('Wrong geometry type!')
        return []

    if symbol in cache:
        return cache[symbol]

    file_name = name_map[symbol]
    file_name = os.path.join(os.path.dirname(__file__), file_name)
    if geometry_type in [GeometryType.SOLID_3D, GeometryType.SOLID_2D]:
        triangles = parse_obj(file_name)
        cache[symbol] = triangles
        return triangles

    lines = open(file_name).readlines()
    vertices_lines = filter(lambda line: line.startswith('v '), lines)
    edges_lines =    filter(lambda line: line.startswith('f '), lines)
    vertices = [map(float, line.split()[1:]) for line in vertices_lines]
    edges_indices = [map(int, line.split()[1:]) for line in edges_lines]
    edges = []
    for i0, i1 in edges_indices:
        v0 = vertices[i0-1]
        v1 = vertices[i1-1]
        edges.append([v0, v1])
    cache[symbol] = edges
    return edges
