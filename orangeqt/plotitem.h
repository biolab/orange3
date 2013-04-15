/*
    This file is part of the plot module for Orange
    Copyright (C) 2011  Miha Čančula <miha@noughmad.eu>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PLOTITEM_H
#define PLOTITEM_H

#include <QtGui/QGraphicsObject>

class Plot;

class PlotItem : public QGraphicsObject
{
    Q_OBJECT
public:
    explicit PlotItem(QGraphicsItem* parent = 0);
    virtual ~PlotItem();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    virtual QRectF boundingRect() const;
    
    virtual QRectF data_rect() const;
    void set_data_rect(const QRectF& dataRect);
    
    virtual void set_graph_transform(const QTransform& transform);
    virtual QTransform graph_transform() const;
    
    virtual void set_zoom_transform(const QTransform& zoom);
    virtual QTransform zoom_transform() const;
    
    virtual void update_properties();
    
    void attach(Plot* graph);
    void detach();
    Plot* plot();
    virtual void register_points();
    
    static QRectF rect_from_data(const QList<double>& x_data, const QList<double>& y_data);
    static void move_item(QGraphicsObject* item, const QPointF& pos, bool animate = true, int duration = 250);
    
    bool is_auto_scale() const;
    void set_auto_scale(bool auto_scale);
    
    bool is_in_background() const;
    void set_in_background(bool bg);
    
    QPair<int, int> axes() const;
    void set_axes(int x_axis, int y_axis);
    
    inline void set_x_axis(int x_axis)
    {
        set_axes(x_axis, axes().second);
    }
    inline void set_y_axis(int y_axis)
    {
        set_axes(axes().first, y_axis);
    }
    
private:
    Q_DISABLE_COPY(PlotItem)
    
    Plot* m_plot;
    QRectF m_dataRect;
    QPair<int, int> m_axes;
    bool m_autoScale;
    QTransform m_graphTransform;
    QTransform m_zoom_transform;
    
    bool m_background;
    
    friend class Plot;
    
};

#endif // PLOTITEM_H
