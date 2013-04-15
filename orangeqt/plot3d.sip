
class Plot3D : QGLWidget {

%TypeHeaderCode
#include "plot3d.h"
%End

public:
    Plot3D(QWidget* parent /TransferThis/ = 0);
    virtual ~Plot3D();

    void set_symbol_geometry(int symbol,
                             int type,
                             const QList<QVector3D>& geometry);

    void set_data(SIP_PYOBJECT,
                  SIP_PYOBJECT,  // int num_examples
                  SIP_PYOBJECT); // int example_size
%MethodCode
    int size = -1;
    float* data = NULL;
    int num_examples = -1;
    int example_size = -1;
    if (!PyInt_Check(a1) || !PyInt_Check(a2))
    {
        PyErr_SetString(PyExc_RuntimeError, "Wrong argument types!");
        return 0;
    }
    num_examples = (int)PyInt_AsLong(a1);
    example_size = (int)PyInt_AsLong(a2);
    bool success = convert_numpy_array_to_native_f(a0, data, size);
    if (!success)
    {
        return 0;
    }
    else if (size != num_examples * example_size)
    {
        delete [] data;
        PyErr_SetString(PyExc_RuntimeError, "Data conversion corrupted!");
        return 0;
    }
    sipCpp->Plot3D::set_data(data, num_examples, example_size);
%End

    void set_data(float* data_array,
                  int num_examples,
                  int example_size);

    void set_valid_data(SIP_PYOBJECT);
%MethodCode
    int size = -1;
    bool* data = NULL;
    bool success = convert_numpy_array_to_native_b(a0, data, size);
    if (!success)
    {
        return 0;
    }
    else if (size != sipCpp->Plot3D::get_num_examples())
    {
        delete [] data;
        PyErr_SetString(PyExc_RuntimeError, "Data conversion corrupted!");
        return 0;
    }
    sipCpp->Plot3D::set_valid_data(data);
%End

    void set_valid_data(bool* valid_data);

    void update_data(int x_index, int y_index, int z_index,
                     int color_index, int symbol_index, int size_index, int label_index,
                     const QList<QColor>& colors, int num_symbols_used,
                     bool x_discrete, bool y_discrete, bool z_discrete, bool use_2d_symbols);

    void draw_data(GLuint shader_id, float alpha_value);
    void draw_data_solid(); // (only draws solid geometry as a performance optimization)

    QList<double> get_min_max_selected(const QList<int>& area,
                                      const QMatrix4x4& mvp,
                                      const QList<int>& viewport,
                                      const QVector3D& plot_scale,
                                      const QVector3D& plot_translation);

    void select_points(const QList<int>& area,
                       const QMatrix4x4& mvp,
                       const QList<int>& viewport,
                       const QVector3D& plot_scale,
                       const QVector3D& plot_translation,
                       Plot::SelectionBehavior behavior = Plot::AddSelection);

    void unselect_all_points();

    QList<bool> get_selected_indices();

    int get_num_examples();
};
