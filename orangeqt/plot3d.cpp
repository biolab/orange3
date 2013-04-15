#include "plot3d.h"

#include <iostream>
#include <limits>
#include <QGLFormat>
#include <numpy/arrayobject.h>

#include "glextensions.h"

Plot3D::Plot3D(QWidget* parent) : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    vbos_generated = false;
    data_array = NULL;
    valid_data = NULL;
    num_examples = 0;
    example_size = 0;
    num_selected_vertices = 0;
    num_unselected_vertices = 0;
    num_edges_vertices = 0;

    // Initialize numpy C-API
    import_array();
}

Plot3D::~Plot3D()
{
    if (vbos_generated)
    {
        glDeleteBuffers(1, &vbo_selected_id);
        glDeleteBuffers(1, &vbo_unselected_id);
        glDeleteBuffers(1, &vbo_edges_id);
    }

    if (data_array != NULL)
        delete [] data_array;

    if (valid_data != NULL)
        delete [] valid_data;
}

void Plot3D::set_symbol_geometry(int symbol, int type, const QList<QVector3D>& geometry)
{
    switch (type)
    {
        case 0:
            geometry_data_2d[symbol] = geometry;
            break;
        case 1:
            geometry_data_3d[symbol] = geometry;
            break;
        case 2:
            geometry_data_edges_2d[symbol] = geometry;
            break;
        case 3:
            geometry_data_edges_3d[symbol] = geometry;
            break;
        default:
            std::cout << "Wrong geometry type!" << std::endl;
    }
}

void Plot3D::set_data(float* data_array, int num_examples, int example_size)
{
    if (this->data_array != NULL)
        delete [] this->data_array; // Clean previous data.

    this->data_array = data_array;  // Data is copied to a fresh array in set_data()
    this->num_examples = num_examples;
    this->example_size = example_size;
    selected_indices = QVector<bool>(num_examples);

    // Load required extensions (OpenGL context should be up by now).
    init_gl_extensions();
}

void Plot3D::set_valid_data(bool* valid_data)
{
    if (this->valid_data != NULL)
        delete [] this->valid_data;

    this->valid_data = valid_data;
}

void Plot3D::update_data(int x_index, int y_index, int z_index,
                         int color_index, int symbol_index, int size_index, int label_index,
                         const QList<QColor>& colors, int num_symbols_used,
                         bool x_discrete, bool y_discrete, bool z_discrete, bool use_2d_symbols)
{
    if (data_array == NULL)
    {
        std::cout << "set_data must be called before update_data!" << std::endl;
        return;
    }

    if (vbos_generated)
    {
        glDeleteBuffers(1, &vbo_selected_id);
        glDeleteBuffers(1, &vbo_unselected_id);
        glDeleteBuffers(1, &vbo_edges_id);
    }

    this->x_index = x_index;
    this->y_index = y_index;
    this->z_index = z_index;

    const float scale = 0.001f;

    float* vbo_selected_data   = new float[num_examples * 144 * 14];
    float* vbo_unselected_data = new float[num_examples * 144 * 14];
    float* vbo_edges_data      = new float[num_examples * 144 * 14];
    float* dests = vbo_selected_data;
    float* destu = vbo_unselected_data;
    float* deste = vbo_edges_data;
    // Sizes in bytes.
    int sib_selected   = 0;
    int sib_unselected = 0;
    int sib_edges      = 0;

    QMap<int, QList<QVector3D> >& geometry =       use_2d_symbols ? geometry_data_2d       : geometry_data_3d;
    QMap<int, QList<QVector3D> >& geometry_edges = use_2d_symbols ? geometry_data_edges_2d : geometry_data_edges_3d;

    for (int index = 0; index < num_examples; ++index)
    {
        if (valid_data != NULL && !valid_data[index]) // Skip invalid examples
            continue;

        float* example = data_array + index*example_size;
        float x_pos = *(example + x_index);
        float y_pos = *(example + y_index);
        float z_pos = *(example + z_index);

        int symbol = 0;
        if (num_symbols_used > 1 && symbol_index > -1)
            symbol = *(example + symbol_index) * num_symbols_used;

        float size = *(example + size_index);
        if (size_index < 0 || size < 0.)
            size = 1.;

        float color_value = *(example + color_index);
        int num_colors = colors.count();
        QColor color;

        if (num_colors > 0)
            color = colors[int(color_value * num_colors)];
        else if (color_index > -1)
            color = QColor(0., 0., int(color_value*255));
        else
            color = QColor(0., 0., 0);

        float*& dest = selected_indices[index] ? dests : destu;

        for (int i = 0; i < geometry[symbol].count(); i += 6) {
            if (selected_indices[index])
                sib_selected += 3*14*4;
            else
                sib_unselected += 3*14*4;

            for (int j = 0; j < 3; ++j)
            {
                // position + first part of the index
                *dest = x_pos; dest++; 
                *dest = y_pos; dest++; 
                *dest = z_pos; dest++; 
                *dest = (float)(index & 0xFF) / 255.f; dest++;

                // offset + second part of the index
                *dest = geometry[symbol][i+j].x()*size*scale; dest++;
                *dest = geometry[symbol][i+j].y()*size*scale; dest++;
                *dest = geometry[symbol][i+j].z()*size*scale; dest++;
                *dest = (float)((index & 0xFF00) >> 8) / 255.f; dest++;

                // color
                *dest = color.redF(); dest++;
                *dest = color.greenF(); dest++;
                *dest = color.blueF(); dest++;

                // normal
                *dest = geometry[symbol][i+3+j].x(); dest++;
                *dest = geometry[symbol][i+3+j].y(); dest++;
                *dest = geometry[symbol][i+3+j].z(); dest++;
            }
        }

        // No need for edges in selected examples (those are drawn fully opaque)
        if (selected_indices[index])
            continue;

        for (int i = 0; i < geometry_edges[symbol].count(); i += 2) {
            sib_edges += 2*14*4;

            for (int j = 0; j < 2; ++j)
            {
                *deste = x_pos; deste++; 
                *deste = y_pos; deste++; 
                *deste = z_pos; deste++; 
                *deste = (float)(index & 0xFF) / 255.f; deste++;

                *deste = geometry_edges[symbol][i+j].x()*size*scale; deste++;
                *deste = geometry_edges[symbol][i+j].y()*size*scale; deste++;
                *deste = geometry_edges[symbol][i+j].z()*size*scale; deste++;
                *deste = (float)((index & 0xFF00) >> 8) / 255.f; deste++;

                *deste = color.redF(); deste++;
                *deste = color.greenF(); deste++;
                *deste = color.blueF(); deste++;

                // Just use offset as the normal for now
                *deste = geometry_edges[symbol][i+j].x(); deste++;
                *deste = geometry_edges[symbol][i+j].y(); deste++;
                *deste = geometry_edges[symbol][i+j].z(); deste++;
            }
        }
    }

    glGenBuffers(1, &vbo_selected_id);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_selected_id);
    glBufferData(GL_ARRAY_BUFFER, sib_selected, vbo_selected_data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete [] vbo_selected_data;

    glGenBuffers(1, &vbo_unselected_id);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_unselected_id);
    glBufferData(GL_ARRAY_BUFFER, sib_unselected, vbo_unselected_data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete [] vbo_unselected_data;

    glGenBuffers(1, &vbo_edges_id);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_edges_id);
    glBufferData(GL_ARRAY_BUFFER, sib_edges, vbo_edges_data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete [] vbo_edges_data;

    num_selected_vertices = sib_selected / (14*4);
    num_unselected_vertices = sib_unselected / (14*4);
    num_edges_vertices = sib_edges / (14*4);

    vbos_generated = true;
}

void Plot3D::draw_data_solid()
{
    if (!vbos_generated)
        return;

    glBindBuffer(GL_ARRAY_BUFFER, vbo_selected_id);
    int vertex_size = 14;
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(0));
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(4*4));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(8*4));
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(11*4));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

    glDrawArrays(GL_TRIANGLES, 0, num_selected_vertices);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_unselected_id);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(0));
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(4*4));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(8*4));
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(11*4));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

    glDrawArrays(GL_TRIANGLES, 0, num_unselected_vertices);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Plot3D::draw_data(GLuint shader_id, float alpha_value)
{
    if (!vbos_generated)
        return;

    // Draw opaque selected examples first.
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_selected_id);
    int vertex_size = 14;
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(0));
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(4*4));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(8*4));
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(11*4));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

    glDrawArrays(GL_TRIANGLES, 0, num_selected_vertices);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);

    // Draw transparent unselected examples (triangles and then edges).
    glUniform2f(glGetUniformLocation(shader_id, "alpha_value"), alpha_value-0.6, alpha_value-0.6);

    glDepthMask(GL_FALSE);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_unselected_id);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(0));
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(4*4));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(8*4));
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(11*4));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

    glDrawArrays(GL_TRIANGLES, 0, num_unselected_vertices);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);

    // Edges
    glDepthMask(GL_TRUE);
    glUniform2f(glGetUniformLocation(shader_id, "alpha_value"), alpha_value, alpha_value);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_edges_id);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(0));
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(4*4));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(8*4));
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(11*4));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

    glDrawArrays(GL_LINES, 0, num_edges_vertices);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

QList<double> Plot3D::get_min_max_selected(const QList<int>& area,
                                           const QMatrix4x4& mvp,
                                           const QList<int>& viewport,
                                           const QVector3D& plot_scale,
                                           const QVector3D& plot_translation)
{
    float x_min = std::numeric_limits<float>::infinity();
    float x_max = -std::numeric_limits<float>::infinity();
    float y_min = x_min;
    float y_max = x_max;
    float z_min = x_min;
    float z_max = x_max;

    bool any_point_selected = false;
    for (int index = 0; index < num_examples; ++index)
    {
        float* example = data_array + index*example_size;
        float x_pos = *(example + x_index);
        float y_pos = *(example + y_index);
        float z_pos = *(example + z_index);

        QVector3D position(x_pos, y_pos, z_pos);
        position += plot_translation;
        position *= plot_scale;

        QVector4D projected = mvp * QVector4D(position, 1.0f);
        projected /= projected.z();
        int winx = viewport[0] + (1 + projected.x()) * viewport[2] / 2;
        int winy = viewport[1] + (1 + projected.y()) * viewport[3] / 2;
        winy = viewport[3] - winy;

        if (winx >= area[0] && winx <= area[0]+area[2] && winy <= area[1]+area[3] && winy >= area[1])
        {
            any_point_selected = true;

            if (x_pos < x_min) x_min = x_pos;
            if (x_pos > x_max) x_max = x_pos;

            if (y_pos < y_min) y_min = y_pos;
            if (y_pos > y_max) y_max = y_pos;

            if (z_pos < z_min) z_min = z_pos;
            if (z_pos > z_max) z_max = z_pos;
        }
    }

    if (any_point_selected)
    {
        QList<double> min_max;
        min_max << x_min << x_max;
        min_max << y_min << y_max;
        min_max << z_min << z_max;
        return min_max;
    }
    else
    {
        QList<double> min_max;
        min_max << 0. << 1.;
        min_max << 0. << 1.;
        min_max << 0. << 1.;
        return min_max;
    }
}

void Plot3D::select_points(const QList<int>& area,
                           const QMatrix4x4& mvp,
                           const QList<int>& viewport,
                           const QVector3D& plot_scale,
                           const QVector3D& plot_translation,
                           Plot::SelectionBehavior behavior)
{
    if (behavior == Plot::ReplaceSelection)
        selected_indices.fill(false);

    for (int index = 0; index < num_examples; ++index)
    {
        float* example = data_array + index*example_size;
        float x_pos = *(example + x_index);
        float y_pos = *(example + y_index);
        float z_pos = *(example + z_index);

        QVector3D position(x_pos, y_pos, z_pos);
        position += plot_translation;
        position *= plot_scale;

        QVector4D projected = mvp * QVector4D(position, 1.0f);
        projected /= projected.z();
        int winx = viewport[0] + (1 + projected.x()) * viewport[2] / 2;
        int winy = viewport[1] + (1 + projected.y()) * viewport[3] / 2;
        winy = viewport[3] - winy;

        if (winx >= area[0] && winx <= area[0]+area[2] && winy <= area[1]+area[3] && winy >= area[1])
        {
            if (behavior == Plot::AddSelection || behavior == Plot::ReplaceSelection)
                selected_indices[index] = true;
            else if (behavior == Plot::RemoveSelection)
                selected_indices[index] = false;
            else if (behavior == Plot::ToggleSelection)
                selected_indices[index] = !selected_indices[index];
        }
    }
}

void Plot3D::unselect_all_points()
{
    selected_indices = QVector<bool>(num_examples);
}

QList<bool> Plot3D::get_selected_indices()
{
    return selected_indices.toList();
}

int Plot3D::get_num_examples()
{
    return num_examples;
}

bool convert_numpy_array_to_native_f(PyObject* in, float*& out_data, int& out_size)
{
    if (!PyArray_Check(in))
    {
        PyErr_SetString(PyExc_RuntimeError, "Unknown object type (must be numpy.array)");
        return false;
    }

    PyObject* array = PyArray_ContiguousFromAny(in, PyArray_FLOAT, 1, 0); // Returns a C-style contiguous and behaved (aligned + writable) array

    if (!array)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to make contiguous array of PyArray_FLOAT");
        return false;
    }

    out_size = PyArray_DIM(array, 0);
    out_data = new float[out_size];
    memcpy(out_data, (float *)PyArray_DATA(array), out_size*sizeof(float));
    Py_DECREF(array);
    return true;
}

bool convert_numpy_array_to_native_b(PyObject* in, bool*& out_data, int& out_size)
{
    if (!PyArray_Check(in))
    {
        PyErr_SetString(PyExc_RuntimeError, "Unknown object type (must be numpy.array)");
        return false;
    }

    PyObject* array = PyArray_ContiguousFromAny(in, PyArray_BOOL, 1, 0);

    if (!array)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to make contiguous array of PyArray_BOOL");
        return false;
    }

    out_size = PyArray_DIM(array, 0);
    out_data = new bool[out_size];
    memcpy(out_data, (bool *)PyArray_DATA(array), out_size*sizeof(bool));
    Py_DECREF(array);
    return true;
}

#include "plot3d.moc"
