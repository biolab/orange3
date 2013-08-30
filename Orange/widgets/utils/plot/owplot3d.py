'''

#################
Plot3D (``owplot3D``)
#################

.. autoclass:: OrangeWidgets.plot.OWPlot3D

'''

import os
import time
from math import sin, cos, pi
import struct

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtOpenGL

from OWDlgs import OWChooseImageSizeDlg
from Orange.utils import deprecated_attribute, deprecated_members

import orangeqt
from .owplotgui import OWPlotGUI
from .owtheme import PlotTheme
from .Orange.widgets.utils.plot.owplot import OWPlot
from .owlegend import OWLegend, OWLegendItem, OWLegendTitle, OWLegendGradient
from .Orange.widgets.utils.plot.owopenglrenderer import OWOpenGLRenderer
from .owconstants import ZOOMING, PANNING, ROTATING

from OWColorPalette import ColorPaletteGenerator

import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.FULL_LOGGING = False
OpenGL.ERROR_ON_COPY = False
from OpenGL.GL import *
from OpenGL.GL.ARB.vertex_array_object import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from ctypes import c_void_p, c_char, c_char_p, POINTER

import numpy

try:
    from itertools import chain

except:
    pass

def vec_div(v1, v2):
    return QVector3D(v1.x() / v2.x(),
                     v1.y() / v2.y(),
                     v1.z() / v2.z())

def lower_bound(value, vec):
    if vec.x() < value:
        vec.setX(value)
    if vec.y() < value:
        vec.setY(value)
    if vec.z() < value:
        vec.setZ(value)
    return vec

def enum(*sequential):
    enums = dict(zip(sequential, range(len(sequential))))
    enums['is_valid'] = lambda self, enum_value: enum_value < len(sequential)
    enums['to_str'] = lambda self, enum_value: sequential[enum_value]
    enums['__len__'] = lambda self: len(sequential)
    return type('Enum', (), enums)()

PlotState = enum('IDLE', 'DRAGGING_LEGEND', 'ROTATING', 'SCALING', 'SELECTING', 'PANNING')

Symbol = enum('RECT', 'TRIANGLE', 'DTRIANGLE', 'CIRCLE', 'LTRIANGLE',
              'DIAMOND', 'WEDGE', 'LWEDGE', 'CROSS', 'XCROSS')

from plot.primitives import get_symbol_geometry, clamp, GeometryType

class OWLegend3D(OWLegend):
    def set_symbol_geometry(self, symbol, geometry):
        if not hasattr(self, '_symbol_geometry'):
            self._symbol_geometry = {}
        self._symbol_geometry[symbol] = geometry

    def _draw_item_background(self, pos, item, color):
        rect = item.rect().normalized().adjusted(pos.x(), pos.y(), pos.x(), pos.y())
        self.widget.renderer.draw_rectangle(
            QVector3D(rect.left(), rect.top(), 0),
            QVector3D(rect.left(), rect.bottom(), 0),
            QVector3D(rect.right(), rect.bottom(), 0),
            QVector3D(rect.right(), rect.top(), 0),
            color=color)

    def _draw_symbol(self, pos, symbol):
        edges = self._symbol_geometry[symbol.symbol()]
        color = symbol.color()
        size = symbol.size() / 2
        for v0, v1 in zip(edges[::2], edges[1::2]):
            x0, y0 = v0.x(), v0.y()
            x1, y1 = v1.x(), v1.y()
            self.widget.renderer.draw_line(
                QVector3D(x0*size + pos.x(), -y0*size + pos.y(), 0),
                QVector3D(x1*size + pos.x(), -y1*size + pos.y(), 0),
                color=color)

    def _paint(self, widget):
        self.widget = widget
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        offset = QPointF(0, 15) # TODO

        for category in self.items:
            items = self.items[category]
            for item in items:
                if isinstance(item, OWLegendTitle):
                    pos = self.pos() + item.pos()
                    self._draw_item_background(pos, item.rect_item, widget._theme.background_color)

                    widget.qglColor(widget._theme.labels_color)
                    pos = self.pos() + item.pos() + item.text_item.pos() + offset
                    widget.renderText(pos.x(), pos.y(), item.text_item.toPlainText(), item.text_item.font())
                elif isinstance(item, OWLegendItem):
                    pos = self.pos() + item.pos()
                    self._draw_item_background(pos, item.rect_item, widget._theme.background_color)

                    widget.qglColor(widget._theme.labels_color)
                    pos = self.pos() + item.pos() + item.text_item.pos() + offset
                    widget.renderText(pos.x(), pos.y(), item.text_item.toPlainText(), item.text_item.font())

                    symbol = item.point_item
                    pos = self.pos() + item.pos() + symbol.pos()
                    self._draw_symbol(pos, symbol)
                elif isinstance(item, OWLegendGradient):
                    pos = self.pos() + item.pos()
                    proxy = lambda: None
                    proxy.rect = lambda: item.rect
                    self._draw_item_background(pos, proxy, widget._theme.background_color)

                    widget.qglColor(widget._theme.labels_color)
                    for label in item.label_items:
                        pos = self.pos() + item.pos() + label.pos() + offset + QPointF(5, 0)
                        widget.renderText(pos.x(), pos.y(), label.toPlainText(), label.font())

                    pos = self.pos() + item.pos() + item.gradient_item.pos()
                    rect = item.gradient_item.rect().normalized().adjusted(pos.x(), pos.y(), pos.x(), pos.y())
                    widget.renderer.draw_rectangle(
                        QVector3D(rect.left(), rect.top(), 0),
                        QVector3D(rect.left(), rect.bottom(), 0),
                        QVector3D(rect.right(), rect.bottom(), 0),
                        QVector3D(rect.right(), rect.top(), 0),
                        QColor(0, 0, 0),
                        QColor(0, 0, 255),
                        QColor(0, 0, 255),
                        QColor(0, 0, 0))

name_map = {
    "saveToFileDirect": "save_to_file_direct",
    "saveToFile" : "save_to_file",
}

@deprecated_members(name_map, wrap_methods=list(name_map.keys()))
class OWPlot3D(orangeqt.Plot3D):
    '''
    The base class behind 3D plots in Orange. Uses OpenGL as its rendering platform.

    **Settings**

        .. attribute:: show_legend

            A boolean controlling whether the legend is displayed or not

        .. attribute:: gui

            An :obj:`.OWPlotGUI` object associated with this graph

    **Data**
        This is the most important part of the Plot3D API. :meth:`set_plot_data` is
        used to (not surprisingly) set the data to draw.
        :meth:`set_features` tells Plot3D how to interpret the data (this method must
        be called after :meth:`set_plot_data` and can be called multiple times).
        :meth:`set_valid_data` optionally informs the plot which examples are invalid and
        should not be drawn. It should be called after set_plot_data, but before set_features.
        This separation permits certain optimizations, e.g. ScatterPlot3D sets data once only (at
        the beginning), later on it calls solely set_features and set_valid_data.

        .. automethod:: set_plot_data

        .. automethod:: set_valid_data

        .. automethod:: set_features

    **Selections**
        There are four possible selection behaviors used for selecting points in OWPlot3D.

        .. data:: AddSelection

            Points are added to the current selection, without affecting currently selected points.

        .. data:: RemoveSelection

            Points are removed from the current selection.

        .. data:: ToggleSelection

            The points' selection state is toggled.

        .. data:: ReplaceSelection

            The current selection is replaced with new one.

        .. automethod:: select_points

        .. automethod:: unselect_all_points

        .. automethod:: get_selected_indices

        .. automethod:: get_min_max_selected

        .. automethod:: set_selection_behavior

    **Callbacks**

        Plot3D provides several callbacks which can be used to perform additional tasks (
        such as drawing custom geometry before/after the data is drawn). For example usage
        see ``OWScatterPlot3D``. Callbacks provided:

            ==============               ==================================================
            Callback                     Event
            --------------               --------------------------------------------------
            auto_send_selection_callback Selection changed.
            mouseover_callback           Mouse cursor moved over a point. Also send example's index.
            before_draw_callback         Right before drawing points (but after
                                         current camera transformations have been computed,
                                         so it's safe to use ``projection``, ``view`` and
                                         ``model``).
            after_draw_callback          Right after points have been drawn.
            ==============               ==================================================

    **Coordinate transformations**

        Data set by ``set_plot_data`` is in what Plot3D calls
        data coordinate system. Plot coordinate system takes into account plot translation
        and scale (together effectively being zoom as well).

        .. automethod:: map_to_plot

        .. automethod:: map_to_data

    **Themes**

        Colors used for data points are specified with two palettes, one for continuous attributes, and one for
        discrete ones.  Both are created by
        :obj:`.OWColorPalette.ColorPaletteGenerator`

        .. attribute:: continuous_palette

            The palette used when point color represents a continuous attribute

        .. attribute:: discrete_palette

            The palette used when point color represents a discrete attribute
    '''
    def __init__(self, parent=None):
        orangeqt.Plot3D.__init__(self, parent)

        # Don't clear background when using QPainter
        self.setAutoFillBackground(False)

        self.camera_distance = 6.
        self.scale_factor = 0.30
        self.rotation_factor = 0.3
        self.zoom_factor = 2000.
        self.yaw = self.pitch = -pi / 4.
        self.panning_factor = 0.8
        self.perspective_near = 0.5
        self.perspective_far = 10.
        self.camera_fov = 14.
        self.update_camera()

        self.show_legend = True
        self._legend = OWLegend3D(self, None)
        self._legend_margin = QRectF(0, 0, 100, 0)
        self._legend_moved = False
        self._legend.set_floating(True)
        self._legend.set_orientation(Qt.Vertical)

        self.use_2d_symbols = False
        self.symbol_scale = 1.
        self.alpha_value = 255

        self._state = PlotState.IDLE
        self._selection = None
        self.selection_behavior = OWPlot.AddSelection

        ## Callbacks
        self.auto_send_selection_callback = None
        self.mouseover_callback = None
        self.before_draw_callback = None
        self.after_draw_callback = None

        self.setMouseTracking(True)
        self._mouse_position = QPoint(0, 0)
        self.invert_mouse_x = False
        self.mouse_sensitivity = 5

        self.clear_plot_transformations()

        self._theme = PlotTheme()
        self._tooltip_fbo_dirty = True
        self._tooltip_win_center = [0, 0]
        self._use_fbos = True

        # If True, do drawing using instancing + geometry shader processing,
        # if False, build VBO every time set_plot_data is called.
        self._use_opengl_3 = False
        self.hide_outside = False
        self.fade_outside = True

        self.data = self.data_array = None
        self.x_index = -1
        self.y_index = -1
        self.z_index = -1
        self.label_index = -1
        self.color_index = -1
        self.symbol_index = -1
        self.size_index = -1
        self.colors = []
        self.num_symbols_used = -1
        self.x_discrete = False
        self.y_discrete = False
        self.z_discrete = False

        self.continuous_palette = ColorPaletteGenerator(numberOfColors=-1)
        self.discrete_palette = ColorPaletteGenerator()

        # An :obj:`.OWPlotGUI` object associated with this plot
        self.gui = OWPlotGUI(self)

    def legend(self):
        '''
            Returns the plot's legend, which is a :obj:`OrangeWidgets.plot.OWLegend`
        '''
        return self._legend

    def initializeGL(self):
        if hasattr(self, '_init_done'):
            return
        self.makeCurrent()
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH) # TODO
        glDisable(GL_CULL_FACE)
        glEnable(GL_MULTISAMPLE)

        # TODO: check hardware for OpenGL 3.x+ support

        self.renderer = OWOpenGLRenderer()

        if self._use_opengl_3:
            self._feedback_generated = False

            self.generating_program = QtOpenGL.QGLShaderProgram()
            self.generating_program.addShaderFromSourceFile(QtOpenGL.QGLShader.Geometry,
                os.path.join(os.path.dirname(__file__), 'generator.gs'))
            self.generating_program.addShaderFromSourceFile(QtOpenGL.QGLShader.Vertex,
                os.path.join(os.path.dirname(__file__), 'generator.vs'))
            varyings = (c_char_p * 5)()
            varyings[:] = ['out_position', 'out_offset', 'out_color', 'out_normal', 'out_index']
            glTransformFeedbackVaryings(self.generating_program.programId(), 5,
                ctypes.cast(varyings, POINTER(POINTER(c_char))), GL_INTERLEAVED_ATTRIBS)

            self.generating_program.bindAttributeLocation('index', 0)

            if not self.generating_program.link():
                print('Failed to link generating shader! Attribute changes may be slow.')
            else:
                print('Generating shader linked.')

            # Upload all symbol geometry into a TBO (texture buffer object), so that generating
            # geometry shader will have access to it. (TBO is easier to use than a texture in this use case).
            geometry_data = []
            symbols_indices = []
            symbols_sizes = []
            for symbol in range(len(Symbol)):
                triangles = get_symbol_geometry(symbol, GeometryType.SOLID_3D)
                symbols_indices.append(len(geometry_data) / 3)
                symbols_sizes.append(len(triangles))
                for tri in triangles:
                    geometry_data.extend(chain(*tri))

            for symbol in range(len(Symbol)):
                triangles = get_symbol_geometry(symbol, GeometryType.SOLID_2D)
                symbols_indices.append(len(geometry_data) / 3)
                symbols_sizes.append(len(triangles))
                for tri in triangles:
                    geometry_data.extend(chain(*tri))

            self.symbols_indices = symbols_indices
            self.symbols_sizes = symbols_sizes

            tbo = glGenBuffers(1)
            glBindBuffer(GL_TEXTURE_BUFFER, tbo)
            glBufferData(GL_TEXTURE_BUFFER, len(geometry_data)*4, numpy.array(geometry_data, 'f'), GL_STATIC_DRAW)
            glBindBuffer(GL_TEXTURE_BUFFER, 0)
            self.symbol_buffer = glGenTextures(1)
            glBindTexture(GL_TEXTURE_BUFFER, self.symbol_buffer)
            glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo) # 3 floating-point components
            glBindTexture(GL_TEXTURE_BUFFER, 0)

            # Generate dummy vertex buffer (points which will be fed to the geometry shader).
            self.dummy_vao = GLuint(0)
            glGenVertexArrays(1, self.dummy_vao)
            glBindVertexArray(self.dummy_vao)
            vertex_buffer_id = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id)
            glBufferData(GL_ARRAY_BUFFER, numpy.arange(50*1000, dtype=numpy.float32), GL_STATIC_DRAW)
            glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 4, c_void_p(0))
            glEnableVertexAttribArray(0)
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            # Specify an output VBO (and VAO)
            self.feedback_vao = feedback_vao = GLuint(0)
            glGenVertexArrays(1, feedback_vao)
            glBindVertexArray(feedback_vao)
            self.feedback_bid = feedback_bid = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, feedback_bid)
            vertex_size = (3+3+3+3+1)*4
            glBufferData(GL_ARRAY_BUFFER, 20*1000*144*vertex_size, c_void_p(0), GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(0))
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(3*4))
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(6*4))
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(9*4))
            glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(12*4))
            glEnableVertexAttribArray(0)
            glEnableVertexAttribArray(1)
            glEnableVertexAttribArray(2)
            glEnableVertexAttribArray(3)
            glEnableVertexAttribArray(4)
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        else:
            # Load symbol geometry and send it to the C++ parent.
            geometry_data = []
            for symbol in range(len(Symbol)):
                triangles = get_symbol_geometry(symbol, GeometryType.SOLID_2D)
                triangles = [QVector3D(*v) for triangle in triangles for v in triangle]
                orangeqt.Plot3D.set_symbol_geometry(self, symbol, 0, triangles)

                triangles = get_symbol_geometry(symbol, GeometryType.SOLID_3D)
                triangles = [QVector3D(*v) for triangle in triangles for v in triangle]
                orangeqt.Plot3D.set_symbol_geometry(self, symbol, 1, triangles)

                edges = get_symbol_geometry(symbol, GeometryType.EDGE_2D)
                edges = [QVector3D(*v) for edge in edges for v in edge]
                orangeqt.Plot3D.set_symbol_geometry(self, symbol, 2, edges)
                self._legend.set_symbol_geometry(symbol, edges)

                edges = get_symbol_geometry(symbol, GeometryType.EDGE_3D)
                edges = [QVector3D(*v) for edge in edges for v in edge]
                orangeqt.Plot3D.set_symbol_geometry(self, symbol, 3, edges)

        self.symbol_program = QtOpenGL.QGLShaderProgram()
        self.symbol_program.addShaderFromSourceFile(QtOpenGL.QGLShader.Vertex,
            os.path.join(os.path.dirname(__file__), 'symbol.vs'))
        self.symbol_program.addShaderFromSourceFile(QtOpenGL.QGLShader.Fragment,
            os.path.join(os.path.dirname(__file__), 'symbol.fs'))

        self.symbol_program.bindAttributeLocation('position', 0)
        self.symbol_program.bindAttributeLocation('offset',   1)
        self.symbol_program.bindAttributeLocation('color',    2)
        self.symbol_program.bindAttributeLocation('normal',   3)

        if not self.symbol_program.link():
            print('Failed to link symbol shader!')
        else:
            print('Symbol shader linked.')

        self.symbol_program_use_2d_symbols = self.symbol_program.uniformLocation('use_2d_symbols')
        self.symbol_program_symbol_scale   = self.symbol_program.uniformLocation('symbol_scale')
        self.symbol_program_alpha_value    = self.symbol_program.uniformLocation('alpha_value')
        self.symbol_program_scale          = self.symbol_program.uniformLocation('scale')
        self.symbol_program_translation    = self.symbol_program.uniformLocation('translation')
        self.symbol_program_hide_outside   = self.symbol_program.uniformLocation('hide_outside')
        self.symbol_program_fade_outside   = self.symbol_program.uniformLocation('fade_outside')
        self.symbol_program_force_color    = self.symbol_program.uniformLocation('force_color')
        self.symbol_program_encode_color   = self.symbol_program.uniformLocation('encode_color')

        format = QtOpenGL.QGLFramebufferObjectFormat()
        format.setAttachment(QtOpenGL.QGLFramebufferObject.Depth)
        self._tooltip_fbo = QtOpenGL.QGLFramebufferObject(256, 256, format)
        if self._tooltip_fbo.isValid():
            print('Tooltip FBO created.')
        else:
            print('Failed to create tooltip FBO! Tooltips disabled.')
            self._use_fbos = False

        self._init_done = True

    def resizeGL(self, width, height):
        pass

    def update_camera(self):
        self.pitch = clamp(self.pitch, -3., -0.1)
        self.camera = QVector3D(
            sin(self.pitch)*cos(self.yaw),
            cos(self.pitch),
            sin(self.pitch)*sin(self.yaw))

    def get_mvp(self):
        '''
        Return current model, view and projection transforms.
        '''
        projection = QMatrix4x4()
        width, height = self.width(), self.height()
        aspect = float(width) / height if height != 0 else 1
        projection.perspective(self.camera_fov, aspect, self.perspective_near, self.perspective_far)

        view = QMatrix4x4()
        view.lookAt(
            self.camera * self.camera_distance,
            QVector3D(0,-0.1, 0),
            QVector3D(0, 1, 0))

        model = QMatrix4x4()
        return model, view, projection

    def set_2D_mode(self):
        '''
        Sets ortho projection and identity modelview transform. A convenience method which
        can be called before doing 2D drawing.
        '''
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width(), self.height(), 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.projection = QMatrix4x4()
        self.projection.ortho(0, self.width(), self.height(), 0, -1, 1)
        self.model = QMatrix4x4()
        self.view = QMatrix4x4()

    def paintEvent(self, event):
        glViewport(0, 0, self.width(), self.height())
        self.qglClearColor(self._theme.background_color)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        model, view, projection = self.get_mvp()
        self.model = model
        self.view = view
        self.projection = projection

        if self.before_draw_callback:
            self.before_draw_callback()

        plot_scale = lower_bound(1e-5, self.plot_scale+self.additional_scale)

        self.symbol_program.bind()
        self.symbol_program.setUniformValue('modelview', self.view * self.model)
        self.symbol_program.setUniformValue('projection', self.projection)
        self.symbol_program.setUniformValue(self.symbol_program_use_2d_symbols, self.use_2d_symbols)
        self.symbol_program.setUniformValue(self.symbol_program_fade_outside,   self.fade_outside)
        self.symbol_program.setUniformValue(self.symbol_program_hide_outside,   self.hide_outside)
        self.symbol_program.setUniformValue(self.symbol_program_encode_color,   False)
        self.symbol_program.setUniformValue(self.symbol_program_symbol_scale,   self.symbol_scale, self.symbol_scale)
        self.symbol_program.setUniformValue(self.symbol_program_alpha_value,    self.alpha_value / 255., self.alpha_value / 255.)
        self.symbol_program.setUniformValue(self.symbol_program_scale,          plot_scale)
        self.symbol_program.setUniformValue(self.symbol_program_translation,    self.plot_translation)
        self.symbol_program.setUniformValue(self.symbol_program_force_color,    0., 0., 0., 0.)

        if self._use_opengl_3 and self._feedback_generated:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            glBindVertexArray(self.feedback_vao)
            glDrawArrays(GL_TRIANGLES, 0, self.num_primitives_generated*3)
            glBindVertexArray(0)
        elif not self._use_opengl_3:
            glDisable(GL_CULL_FACE)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            orangeqt.Plot3D.draw_data(self, self.symbol_program.programId(), self.alpha_value / 255.)

        self.symbol_program.release()

        self._draw_labels()

        if self.after_draw_callback:
            self.after_draw_callback()

        if self._tooltip_fbo_dirty:
            self._tooltip_fbo.bind()
            glClearColor(1, 1, 1, 1)
            glClearDepth(1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDisable(GL_BLEND)
            glEnable(GL_DEPTH_TEST)

            glViewport(-self._mouse_position.x()+128, -(self.height()-self._mouse_position.y())+128, self.width(), self.height())
            self._tooltip_win_center = [self._mouse_position.x(), self._mouse_position.y()]

            self.symbol_program.bind()
            self.symbol_program.setUniformValue(self.symbol_program_encode_color, True)

            if self._use_opengl_3 and self._feedback_generated:
                glBindVertexArray(self.feedback_vao)
                glDrawArrays(GL_TRIANGLES, 0, self.num_primitives_generated*3)
                glBindVertexArray(0)
            elif not self._use_opengl_3:
                orangeqt.Plot3D.draw_data_solid(self)
            self.symbol_program.release()
            self._tooltip_fbo.release()
            self._tooltip_fbo_dirty = False
            glViewport(0, 0, self.width(), self.height())

        self._draw_helpers()

        if self.show_legend:
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width(), self.height(), 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glDisable(GL_BLEND)

            self._legend._paint(self)

        self.swapBuffers()

    def _draw_labels(self):
        if self.label_index < 0:
            return

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.projection.data(), dtype=float))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.view.data(), dtype=float))
        glMultMatrixd(numpy.array(self.model.data(), dtype=float))

        self.qglColor(self._theme.labels_color)
        for example in self.data.transpose():
            x = example[self.x_index]
            y = example[self.y_index]
            z = example[self.z_index]
            label = example[self.label_index]
            x, y, z = self.map_to_plot(QVector3D(x, y, z))
            # TODO
            #if isinstance(label, str):
                #self.renderText(x,y,z, label, font=self._theme.labels_font)
            #else:
            self.renderText(x,y,z, ('%.3f' % label).rstrip('0').rstrip('.'),
                            font=self._theme.labels_font)

    def _draw_helpers(self):
        glEnable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

        projection = QMatrix4x4()
        projection.ortho(0, self.width(), self.height(), 0, -1, 1)
        model = view = QMatrix4x4()

        self.renderer.set_transform(model, view, projection)

        if self._state == PlotState.SCALING:
            x, y = self._mouse_position.x(), self._mouse_position.y()
            self.renderer.draw_triangle(QVector3D(x-5, y-30, 0),
                                        QVector3D(x+5, y-30, 0),
                                        QVector3D(x, y-40, 0),
                                        color=self._theme.helpers_color)
            self.renderer.draw_line(QVector3D(x, y, 0),
                                    QVector3D(x, y-30, 0),
                                    color=self._theme.helpers_color)
            self.renderer.draw_triangle(QVector3D(x-5, y-10, 0),
                                        QVector3D(x+5, y-10, 0),
                                        QVector3D(x, y, 0),
                                        color=self._theme.helpers_color)

            self.renderer.draw_triangle(QVector3D(x+10, y, 0),
                                        QVector3D(x+20, y-5, 0),
                                        QVector3D(x+20, y+5, 0),
                                        color=self._theme.helpers_color)
            self.renderer.draw_line(QVector3D(x+10, y, 0),
                                    QVector3D(x+40, y, 0),
                                    color=self._theme.helpers_color)
            self.renderer.draw_triangle(QVector3D(x+50, y, 0),
                                        QVector3D(x+40, y-5, 0),
                                        QVector3D(x+40, y+5, 0),
                                        color=self._theme.helpers_color)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width(), self.height(), 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            self.renderText(x, y-50, 'Scale y axis', font=self._theme.labels_font)
            self.renderText(x+60, y+3, 'Scale x and z axes', font=self._theme.labels_font)
        elif self._state == PlotState.SELECTING:
            internal_color = QColor(168, 202, 236, 50)
            self.renderer.draw_rectangle(QVector3D(self._selection.left(), self._selection.top(), 0),
                                         QVector3D(self._selection.right(), self._selection.top(), 0),
                                         QVector3D(self._selection.right(), self._selection.bottom(), 0),
                                         QVector3D(self._selection.left(), self._selection.bottom(), 0),
                                         color=internal_color)

            border_color = QColor(51, 153, 255, 192)
            self.renderer.draw_line(QVector3D(self._selection.left(), self._selection.top(), 0),
                                    QVector3D(self._selection.right(), self._selection.top(), 0),
                                    border_color, border_color)
            self.renderer.draw_line(QVector3D(self._selection.right(), self._selection.top(), 0),
                                    QVector3D(self._selection.right(), self._selection.bottom(), 0),
                                    border_color, border_color)
            self.renderer.draw_line(QVector3D(self._selection.right(), self._selection.bottom(), 0),
                                    QVector3D(self._selection.left(), self._selection.bottom(), 0),
                                    border_color, border_color)
            self.renderer.draw_line(QVector3D(self._selection.left(), self._selection.bottom(), 0),
                                    QVector3D(self._selection.left(), self._selection.top(), 0),
                                    border_color, border_color)

    def set_features(self,
            x_index, y_index, z_index,
            color_index, symbol_index, size_index, label_index,
            colors, num_symbols_used,
            x_discrete, y_discrete, z_discrete):
        '''
        Explains to the plot how to interpret the data set by :meth:`set_plot_data`. Its arguments
        are indices (must be less than the size of an example) into the dataset (each one
        specifies a column). Additionally, it accepts a list of colors (when color is a discrete
        attribute), a value specifying how many different symbols are needed to display the data and
        information whether or not positional data is discrete.

        .. note:: This function does not add items to the legend automatically.
                  You will have to add them yourself with :meth:`.OWLegend.add_item`.

        :param x_index: Index (column) of the x coordinate.
        :type int

        :param y_index: Index (column) of the y coordinate.
        :type int

        :param z_index: Index (column) of the z coordinate.
        :type int

        :param color_index: Index (column) of the color attribute.
        :type int

        :param symbol_index: Index (column) of the symbol attribute.
        :type int

        :param size_index: Index (column) of the size attribute.
        :type int

        :param label_index: Index (column) of the label attribute.
        :type int

        :param colors: List of colors used for symbols. When color is a discrete attribute,
            this list should be empty. You should make sure the number of colors in this list
            equals the number of unique values in the color attribute.
        :type list of QColor

        :param num_symbols_used: Specifies the number of unique values in the symbol attribute.
            Must be -1 if all points are to use the same symbol.
        :type int

        :param x_discrete: Specifies whether or not x coordinate is discrete.
        :type bool

        :param y_discrete: Specifies whether or not y coordinate is discrete.
        :type bool

        :param z_discrete: Specifies whether or not z coordinate is discrete.
        :type bool
        '''
        if self.data == None:
            print('Error: set_plot_data has not been called yet!')
            return
        start = time.time()
        self.makeCurrent()
        self.x_index = x_index
        self.y_index = y_index
        self.z_index = z_index
        self.color_index = color_index
        self.symbol_index = symbol_index
        self.size_index = size_index
        self.colors = colors
        self.num_symbols_used = num_symbols_used
        self.x_discrete = x_discrete
        self.y_discrete = y_discrete
        self.z_discrete = z_discrete
        self.label_index = label_index

        if self.num_examples > 10*1000:
            self.use_2d_symbols = True

        if self._use_opengl_3:
            # Re-run generating program (geometry shader), store
            # results through transform feedback into a VBO on the GPU.
            self.generating_program.bind()
            self.generating_program.setUniformValue('x_index', x_index)
            self.generating_program.setUniformValue('y_index', y_index)
            self.generating_program.setUniformValue('z_index', z_index)
            self.generating_program.setUniformValue('x_discrete', x_discrete)
            self.generating_program.setUniformValue('y_discrete', y_discrete)
            self.generating_program.setUniformValue('z_discrete', z_discrete)
            self.generating_program.setUniformValue('color_index', color_index)
            self.generating_program.setUniformValue('symbol_index', symbol_index)
            self.generating_program.setUniformValue('size_index', size_index)
            self.generating_program.setUniformValue('use_2d_symbols', self.use_2d_symbols)
            self.generating_program.setUniformValue('example_size', self.example_size)
            self.generating_program.setUniformValue('num_colors', len(colors))
            self.generating_program.setUniformValue('num_symbols_used', num_symbols_used)
            # TODO: colors is list of QColor
            glUniform3fv(glGetUniformLocation(self.generating_program.programId(), 'colors'),
                len(colors), numpy.array(colors, 'f').ravel())
            glUniform1iv(glGetUniformLocation(self.generating_program.programId(), 'symbols_sizes'),
                len(Symbol)*2, numpy.array(self.symbols_sizes, dtype='i'))
            glUniform1iv(glGetUniformLocation(self.generating_program.programId(), 'symbols_indices'),
                len(Symbol)*2, numpy.array(self.symbols_indices, dtype='i'))

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_BUFFER, self.symbol_buffer)
            self.generating_program.setUniformValue('symbol_buffer', 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_BUFFER, self.data_buffer)
            self.generating_program.setUniformValue('data_buffer', 1)

            qid = glGenQueries(1)
            glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, qid)
            glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, self.feedback_bid)
            glEnable(GL_RASTERIZER_DISCARD)
            glBeginTransformFeedback(GL_TRIANGLES)

            glBindVertexArray(self.dummy_vao)
            glDrawArrays(GL_POINTS, 0, self.num_examples)

            glEndTransformFeedback()
            glDisable(GL_RASTERIZER_DISCARD)

            glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN)
            self.num_primitives_generated = glGetQueryObjectuiv(qid, GL_QUERY_RESULT)
            glBindVertexArray(0)
            self._feedback_generated = True
            print(('Num generated primitives: ' + str(self.num_primitives_generated)))

            self.generating_program.release()
            glActiveTexture(GL_TEXTURE0)
            print(('Generation took ' + str(time.time()-start) + ' seconds'))
        else:
            orangeqt.Plot3D.update_data(self, x_index, y_index, z_index,
                color_index, symbol_index, size_index, label_index,
                colors, num_symbols_used,
                x_discrete, y_discrete, z_discrete, self.use_2d_symbols)
            print(('Data processing took ' + str(time.time() - start) + ' seconds'))

        self.update()

    def set_plot_data(self, data, subset_data=None):
        '''
        Sets the data to be drawn. Data is expected to be scaled already (see ``OWScatterPlot3D`` for example).

        :param data: Data
        :type data: numpy array
        '''
        self.makeCurrent()
        self.data = data
        self.data_array = numpy.array(data.transpose().flatten(), dtype=numpy.float32)
        self.example_size = len(data)
        self.num_examples = len(data[0])

        if self._use_opengl_3:
            tbo = glGenBuffers(1)
            glBindBuffer(GL_TEXTURE_BUFFER, tbo)
            glBufferData(GL_TEXTURE_BUFFER, len(self.data_array)*4, self.data_array, GL_STATIC_DRAW)
            glBindBuffer(GL_TEXTURE_BUFFER, 0)

            self.data_buffer = glGenTextures(1)
            glBindTexture(GL_TEXTURE_BUFFER, self.data_buffer)
            GL_R32F = 0x822E
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, tbo)
            glBindTexture(GL_TEXTURE_BUFFER, 0)
        else:
            orangeqt.Plot3D.set_data(self, self.data_array,
                                     self.num_examples,
                                     self.example_size)

    def set_valid_data(self, valid_data):
        '''
        Specifies which examples are invalid and should not be displayed.

        :param valid_data: List of booleans: true for valid indices, false otherwise
        :type valid_data: list of bool
        '''
        self.valid_data = numpy.array(valid_data, dtype=bool)
        orangeqt.Plot3D.set_valid_data(self, self.valid_data)

    def set_new_zoom(self, min, max):
        '''
        Specifies new zoom in data coordinates. Zoom is provided in form of plot translation
        and scale, not camera transformation. This might not be what you want (``OWLinProj3D``
        disables this behavior for example). Plot3D remembers translation and scale.
        ``zoom_out`` can be use to restore the previous zoom level.

        :param min: Lower left corner of the new zoom volume.
        :type QVector3D

        :param max: Upper right corner of the new zoom volume.
        :type QVector3D
        '''
        self._zoom_stack.append((self.plot_scale, self.plot_translation))
        center = (max + min) / 2.
        new_translation = -center
        self._zoomed_size = max-min + QVector3D(1e-5, 1e-5, 1e-5)
        new_scale = vec_div(QVector3D(1, 1, 1), self._zoomed_size)
        self._animate_new_scale_translation(new_scale, new_translation)

    def zoom_out(self):
        '''
        Restores previous zoom level.
        '''
        if len(self._zoom_stack) < 1:
            new_translation = QVector3D(-0.5, -0.5, -0.5)
            new_scale = QVector3D(1, 1, 1)
        else:
            new_scale, new_translation = self._zoom_stack.pop()
        self._animate_new_scale_translation(new_scale, new_translation)
        self._zoomed_size = vec_div(QVector3D(1, 1, 1), new_scale)

    def _animate_new_scale_translation(self, new_scale, new_translation, num_steps=10):
        translation_step = (new_translation - self.plot_translation) / float(num_steps)
        scale_step = (new_scale - self.plot_scale) / float(num_steps)
        # Animate zooming: translate first for a number of steps,
        # then scale. Make sure it doesn't take too long.
        start = time.time()
        for i in range(num_steps):
            if time.time() - start > 1.:
                self.plot_translation = new_translation
                break
            self.plot_translation = self.plot_translation + translation_step
            self.repaint()
        for i in range(num_steps):
            if time.time() - start > 1.:
                self.plot_scale = new_scale
                break
            self.plot_scale = self.plot_scale + scale_step
            self.repaint()

    def save_to_file(self, extraButtons=[]):
        print('Save to file called!')
        sizeDlg = OWChooseImageSizeDlg(self, extraButtons, parent=self)
        sizeDlg.exec_()

    def save_to_file_direct(self, fileName, size=None):
        sizeDlg = OWChooseImageSizeDlg(self)
        sizeDlg.saveImage(fileName, size)

    def map_to_plot(self, point):
        '''
        Maps ``point`` to plot coordinates (applies current translation and scale).
        ``point`` is assumed to be in data coordinates.

        :param point: Location in space
        :type QVector3D
        '''
        plot_scale = lower_bound(1e-5, self.plot_scale+self.additional_scale)
        point = (point + self.plot_translation) * plot_scale
        return point

    def map_to_data(self, point):
        '''
        Maps ``point`` to data coordinates (applies inverse of the current translation and scale).
        ``point`` is assumed to be in plot coordinates.

        :param point: Location in space
        :type QVector3D
        '''
        plot_scale = lower_bound(1e-5, self.plot_scale+self.additional_scale)
        point = vec_div(point, plot_scale) - self.plot_translation
        return point

    def get_min_max_selected(self, area):
        '''
        Returns min/max x/y/z coordinate values of currently selected points.

        :param area: Rectangular area.
        :type QRect
        '''
        viewport = [0, 0, self.width(), self.height()]
        area = [min(area.left(), area.right()), min(area.top(), area.bottom()), abs(area.width()), abs(area.height())]
        min_max = orangeqt.Plot3D.get_min_max_selected(self, area, self.projection * self.view * self.model,
                                                       viewport, self.plot_scale, self.plot_translation)
        return min_max

    def get_selected_indices(self):
        '''
        Returns indices of currently selected points (examples).
        '''
        return orangeqt.Plot3D.get_selected_indices(self)

    def select_points(self, area, behavior):
        '''
        Selects all points inside volume specified by rectangular area and current camera transform
        using selection ``behavior``.

        :param area: Rectangular area.
        :type QRect

        :param behavior: :data:`AddSelection`, :data:`RemoveSelection`, :data:`ToggleSelection` or :data:`ReplaceSelection`
        :type behavior: int
        '''
        viewport = [0, 0, self.width(), self.height()]
        area = [min(area.left(), area.right()), min(area.top(), area.bottom()), abs(area.width()), abs(area.height())]
        orangeqt.Plot3D.select_points(self, area, self.projection * self.view * self.model,
                                      viewport, self.plot_scale, self.plot_translation,
                                      behavior)
        orangeqt.Plot3D.update_data(self, self.x_index, self.y_index, self.z_index,
                                    self.color_index, self.symbol_index, self.size_index, self.label_index,
                                    self.colors, self.num_symbols_used,
                                    self.x_discrete, self.y_discrete, self.z_discrete, self.use_2d_symbols)

    def unselect_all_points(self):
        '''
        Unselects everything.
        '''
        orangeqt.Plot3D.unselect_all_points(self)
        orangeqt.Plot3D.update_data(self, self.x_index, self.y_index, self.z_index,
                                    self.color_index, self.symbol_index, self.size_index, self.label_index,
                                    self.colors, self.num_symbols_used,
                                    self.x_discrete, self.y_discrete, self.z_discrete, self.use_2d_symbols)
        self.update()

    def set_selection_behavior(self, behavior):
        '''
        Sets selection behavior.

        :param behavior: :data:`AddSelection`, :data:`RemoveSelection`, :data:`ToggleSelection` or :data:`ReplaceSelection`
        :type behavior: int
        '''
        self.selection_behavior = behavior

    def send_selection(self):
        if self.auto_send_selection_callback:
            self.auto_send_selection_callback()

    def mousePressEvent(self, event):
        pos = self._mouse_position = event.pos()
        buttons = event.buttons()

        self._selection = None

        if buttons & Qt.LeftButton:
            legend_pos = self._legend.pos()
            lx, ly = legend_pos.x(), legend_pos.y()
            if self.show_legend and self._legend.boundingRect().adjusted(lx, ly, lx, ly).contains(pos.x(), pos.y()):
                event.scenePos = lambda: QPointF(pos)
                self._legend.mousePressEvent(event)
                self.setCursor(Qt.ClosedHandCursor)
                self._state = PlotState.DRAGGING_LEGEND
            elif self.state == PANNING:
                self._state = PlotState.PANNING
            elif self.state == ROTATING or QApplication.keyboardModifiers() & Qt.ShiftModifier:
                self._state = PlotState.ROTATING
            else:
                self._state = PlotState.SELECTING
                self._selection = QRect(pos.x(), pos.y(), 0, 0)
        elif buttons & Qt.RightButton:
            if QApplication.keyboardModifiers() & Qt.ShiftModifier:
                self._state = PlotState.SCALING
                self.scaling_init_pos = self._mouse_position
                self.additional_scale = QVector3D(0, 0, 0)
            else:
                self.zoom_out()
            self.update()
        elif buttons & Qt.MiddleButton:
            if QApplication.keyboardModifiers() & Qt.ShiftModifier:
                self._state = PlotState.PANNING
            else:
                self._state = PlotState.ROTATING

    def _check_mouseover(self, pos):
        if self.mouseover_callback != None and self._state == PlotState.IDLE:
            if abs(pos.x() - self._tooltip_win_center[0]) > 100 or\
               abs(pos.y() - self._tooltip_win_center[1]) > 100:
                self._tooltip_fbo_dirty = True
                self.update()
            # Use pixel-color-picking to read example index under mouse cursor (also called ID rendering).
            self._tooltip_fbo.bind()
            value = glReadPixels(pos.x() - self._tooltip_win_center[0] + 128,
                                 self._tooltip_win_center[1] - pos.y() + 128,
                                 1, 1,
                                 GL_RGBA,
                                 GL_UNSIGNED_BYTE)
            self._tooltip_fbo.release()
            value = struct.unpack('I', value)[0]
            # Check if value is less than 4294967295 (
            # the highest 32-bit unsigned integer) which
            # corresponds to white background in color-picking buffer.
            if value < 4294967295:
                self.mouseover_callback(value)

    def mouseMoveEvent(self, event):
        pos = event.pos()

        self._check_mouseover(pos)

        dx = pos.x() - self._mouse_position.x()
        dy = pos.y() - self._mouse_position.y()

        if self.invert_mouse_x:
            dx = -dx

        if self._state == PlotState.SELECTING:
            self._selection.setBottomRight(pos)
        elif self._state == PlotState.DRAGGING_LEGEND:
            event.scenePos = lambda: QPointF(pos)
            self._legend.mouseMoveEvent(event)
        elif self._state == PlotState.ROTATING:
            self.yaw += (self.mouse_sensitivity / 5.) * dx / (self.rotation_factor*self.width())
            self.pitch += (self.mouse_sensitivity / 5.) * dy / (self.rotation_factor*self.height())
            self.update_camera()
        elif self._state == PlotState.PANNING:
            right_vec = QVector3D.crossProduct(self.camera, QVector3D(0, 1, 0)).normalized()
            up_vec = QVector3D.crossProduct(right_vec, self.camera).normalized()
            right_vec.setX(right_vec.x() * dx / (self.width() * self.plot_scale.x() * self.panning_factor))
            right_vec.setZ(right_vec.z() * dx / (self.width() * self.plot_scale.z() * self.panning_factor))
            right_vec.setX(right_vec.x() * (self.mouse_sensitivity / 5.))
            right_vec.setZ(right_vec.z() * (self.mouse_sensitivity / 5.))
            up_scale = self.height() * self.plot_scale.y() * self.panning_factor
            self.plot_translation -= right_vec + up_vec * (dy / up_scale) * (self.mouse_sensitivity / 5.)
        elif self._state == PlotState.SCALING:
            dx = pos.x() - self.scaling_init_pos.x()
            dy = pos.y() - self.scaling_init_pos.y()
            dx /= self.scale_factor * self.width()
            dy /= self.scale_factor * self.height()
            dy /= float(self._zoomed_size.y())
            dx *= self.mouse_sensitivity / 5.
            dy *= self.mouse_sensitivity / 5.
            right_vec = QVector3D.crossProduct(self.camera, QVector3D(0, 1, 0)).normalized()
            self.additional_scale = QVector3D(-dx * abs(right_vec.x()) / float(self._zoomed_size.x()),
                                               dy,
                                              -dx * abs(right_vec.z()) / float(self._zoomed_size.z()))
        elif self._state == PlotState.IDLE:
            legend_pos = self._legend.pos()
            lx, ly = legend_pos.x(), legend_pos.y()
            if self.show_legend and self._legend.boundingRect().adjusted(lx, ly, lx, ly).contains(pos.x(), pos.y()):
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.unsetCursor()

        self._mouse_position = pos
        self.update()

    def mouseReleaseEvent(self, event):
        if self._state == PlotState.DRAGGING_LEGEND:
            self._legend.mouseReleaseEvent(event)
        if self._state == PlotState.SCALING:
            self.plot_scale = lower_bound(1e-5, self.plot_scale+self.additional_scale)
            self.additional_scale = QVector3D(0, 0, 0)
            self._state = PlotState.IDLE
        elif self._state == PlotState.SELECTING:
            self._selection.setBottomRight(event.pos())
            if self.state == ZOOMING: # self.state is actually set by OWPlotGUI (different from self._state!)
                min_max = self.get_min_max_selected(self._selection)
                x_min, x_max, y_min, y_max, z_min, z_max = min_max
                min, max = QVector3D(x_min, y_min, z_min), QVector3D(x_max, y_max, z_max)
                self.set_new_zoom(min, max)
            else:
                self.select_points(self._selection, self.selection_behavior)
                if self.auto_send_selection_callback:
                    self.auto_send_selection_callback()

        self._tooltip_fbo_dirty = True
        self.unsetCursor()
        self._state = PlotState.IDLE
        self.update()

    def wheelEvent(self, event):
        if event.orientation() == Qt.Vertical:
            delta = 1 + event.delta() / self.zoom_factor
            self.plot_scale *= delta
            self._tooltip_fbo_dirty = True
            self.update()

    def notify_legend_moved(self, pos):
        self._legend.set_floating(True, pos)
        self._legend.set_orientation(Qt.Vertical)

    def get_theme(self):
        return self._theme

    def set_theme(self, value):
        self._theme = value
        self.update()

    theme = pyqtProperty(PlotTheme, get_theme, set_theme)

    def color(self, role, group = None):
        if group:
            return self.palette().color(group, role)
        else:
            return self.palette().color(role)

    def set_palette(self, p):
        '''
        Sets the plot palette to ``p``.

        :param p: The new color palette
        :type p: :obj:`.QPalette`
        '''
        self.setPalette(p)
        self.update()

    def show_tooltip(self, text):
        x, y = self._mouse_position.x(), self._mouse_position.y()
        QToolTip.showText(self.mapToGlobal(QPoint(x, y)), text, self, QRect(x-3, y-3, 6, 6))

    def clear(self):
        '''
        Clears the plot (legend) but remembers plot transformations (zoom, scale, translation).
        '''
        self._legend.clear()
        self._tooltip_fbo_dirty = True
        self._feedback_generated = False

    def clear_plot_transformations(self):
        '''
        Forgets plot transformations (zoom, scale, translation).
        '''
        self._zoom_stack = []
        self._zoomed_size = QVector3D(1, 1, 1)
        self.plot_translation = QVector3D(-0.5, -0.5, -0.5)
        self.plot_scale = QVector3D(1, 1, 1)
        self.additional_scale = QVector3D(0, 0, 0)

    contPalette = deprecated_attribute('contPalette', 'continuous_palette')
    discPalette = deprecated_attribute('discPalette', 'discrete_palette')
    showLegend = deprecated_attribute('showLegend', 'show_legend')

if __name__ == "__main__":
    # TODO
    pass
