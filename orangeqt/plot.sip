// QPair<int, int> is implemented as a Python 2-element tuple.
%MappedType QPair<double, double> /DocType="tuple-of-float-float"/
{
%TypeHeaderCode
#include <QtCore/QPair>
%End

%ConvertFromTypeCode
    // Create the tuple.
    return Py_BuildValue((char *)"dd", sipCpp->first, sipCpp->second);
%End

%ConvertToTypeCode
    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
        return (PyTuple_Size(sipPy) == 2);
    
    QPair<double,double>* qp = new QPair<double,double>();

    PyArg_ParseTuple(sipPy, "dd", &qp->first, &qp->second);
    *sipCppPtr = qp;
    
    return sipGetState(sipTransferObj);
%End
};

class Plot : QGraphicsView {

%TypeHeaderCode
#include "plot.h"
%End

public:
    enum SelectionBehavior
    {
        AddSelection,
        RemoveSelection,
        ToggleSelection,
        ReplaceSelection
    };
    
    Plot(QWidget* parent /TransferThis/ = 0);
    virtual ~Plot();
    
    virtual void replot() = 0;
    virtual void start_progress() = 0;
    virtual void set_progress(int done, int total) = 0;
    virtual void end_progress() = 0;
    
    void add_item(PlotItem* item /Transfer/);
    void remove_item(PlotItem* item /TransferBack/);
    QList<PlotItem*> plot_items();
    
    void set_item_in_background(PlotItem* item /Transfer/, bool bg);
    
    QRectF data_rect_for_axes(int xAxis, int yAxis);
    QPair< double, double > bounds_for_axis(int axis);
    
    void set_graph_rect(const QRectF rect);
    void set_zoom_transform(const QTransform& zoom);

    void set_dirty();

    void select_points(const QRectF& rect, SelectionBehavior behavior = AddSelection);
    void select_points(const QPolygonF& area, SelectionBehavior behavior = AddSelection);
    void select_points(const Data& data, SelectionBehavior behavior = AddSelection);
    void unselect_all_points();
    
    void mark_points(const QRectF& rect, SelectionBehavior behavior = AddSelection);
    void mark_points(const QPolygonF& area, SelectionBehavior behavior = AddSelection);
    void mark_points(const Data& data, SelectionBehavior behavior = AddSelection);
    void unmark_all_points();
    
    void selected_to_marked();
    void marked_to_selected();
    
    QList< bool > selected_points(const QList< double > x_data, const QList< double > y_data);
    QList< Point* > selected_points();
    QList< Point*> marked_points();
        
    Point* nearest_point(const QPointF& pos);    
    Point* point_at(const DataPoint& pos);
    Point* selected_point_at(const DataPoint& pos);
    QList<Point*> all_points();
    
    void add_point(Point* point, PlotItem* parent);
    void add_points(const QList<Point*>& items, PlotItem* parent);
    void remove_point(Point* point, PlotItem* parent);
    void remove_all_points(PlotItem* parent);
    
    void move_selected_points(const DataPoint& d);
    
    bool animate_points;

signals:
    void selection_changed();
    void marked_points_changed();
    void point_hovered(Point* point);
    void point_rightclicked(Point* point);
    
protected:
    void set_clean();
    bool is_dirty();
};
