class PlotItem : QGraphicsObject {

%TypeHeaderCode
#include "plotitem.h"
%End

public:
    PlotItem(QGraphicsItem* parent /TransferThis/ = 0);
    virtual ~PlotItem();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    virtual QRectF boundingRect() const;
    
    virtual QRectF data_rect() const;
    void set_data_rect(const QRectF& data_rect);
    
    virtual void set_graph_transform(const QTransform& transform);
    virtual QTransform graph_transform() const;

    virtual void set_zoom_transform(const QTransform& zoom);
    virtual QTransform zoom_transform() const;

    virtual void update_properties();
    
    void attach(Plot* graph /TransferThis/);
    void detach();
%MethodCode
        sipTransferBack(sipGetPyObject(sipCpp, sipType_PlotItem));
        sipCpp->detach();
%End

    Plot* plot();
    
    static QRectF rect_from_data(const QList<double>& x_data, const QList<double>& y_data);
    static void move_item(QGraphicsObject* item, const QPointF& pos, bool animate = true, int duration = 250);
    
    bool is_auto_scale() const;
    void set_auto_scale(bool auto_scale);
    
    bool is_in_background() const;
    void set_in_background(bool bg);
    
    QPair<int, int> axes() const;
    void set_axes(int x_axis, int y_axis);
    
    void set_x_axis(int x_axis);
    void set_y_axis(int y_axis);

};