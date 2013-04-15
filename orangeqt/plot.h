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

#ifndef PLOT_H
#define PLOT_H

#include <QtGui/QGraphicsView>
#include <QtCore/QHash>
#include <QtCore/QMap>

#include "curve.h"

class Point;
class PlotItem;

class Plot : public QGraphicsView
{
    Q_OBJECT
public:
    enum SelectionBehavior
    {
        AddSelection,
        RemoveSelection,
        ToggleSelection,
        ReplaceSelection
    };
    
    typedef QSet<DataPoint> PointSet;
    typedef QMultiHash<DataPoint, Point*> PointHash;

    
    explicit Plot(QWidget* parent = 0);
    virtual ~Plot();
    
    virtual void replot() = 0;
    virtual void start_progress() = 0;
    virtual void set_progress(int done, int total) = 0;
    virtual void end_progress() = 0;
    
    void add_item(PlotItem* item);
    void remove_item(PlotItem* item);
    void set_item_in_background(PlotItem* item, bool bg);

    QRectF data_rect_for_axes(int x_axis, int y_axis);
    QPair< double, double > bounds_for_axis(int axis);
    
    QList<PlotItem*> plot_items();
    
    void set_graph_rect(const QRectF rect);
    void set_zoom_transform(const QTransform& zoom);
    
    void set_dirty();
    
    void select_points(const QRectF& rect, SelectionBehavior behavior = AddSelection);
    void select_points(const QPolygonF& area, SelectionBehavior behavior = AddSelection);
    void select_points(const Data& data, SelectionBehavior behavior = AddSelection);
    
    void mark_points(const QRectF& rect, SelectionBehavior behavior = AddSelection);
    void mark_points(const QPolygonF& area, SelectionBehavior behavior = AddSelection);
    void mark_points(const Data& data, SelectionBehavior behavior = AddSelection);
    
    /**
     * For each point defined with @p x_data and @p y_data, this function checks whether such a point is selected. 
     * This function is precise, so you have to supply it with precisely the same data as the curves that 
     * created the points
     **/
    QList< bool > selected_points(const QList< double > x_data, const QList< double > y_data);
    QList< Point*> selected_points();
    QList< Point*> marked_points();
    
    Point* nearest_point(const QPointF& pos);    
    Point* point_at(const DataPoint& pos);
    Point* selected_point_at(const DataPoint& pos);
    QList<Point*> all_points();

    void add_point(Point* point, PlotItem* parent);
    void add_points(const QList<Point*>& items, PlotItem* parent);
    void remove_point(Point* point, PlotItem* parent);
    void remove_all_points(PlotItem* parent);
    
    void unselect_all_points();
    void unmark_all_points();
    void selected_to_marked();
    void marked_to_selected();
    
    void move_selected_points(const DataPoint& d);

    bool animate_points;
    
    void emit_marked_points_changed();
    void emit_selection_changed();

signals:
    void selection_changed();
    void marked_points_changed();
    void point_hovered(Point* point);
    void point_rightclicked(Point* point);
    
protected:
    void set_clean();
    bool is_dirty();
    
private:    
    QList<PlotItem*> m_items;
    bool m_dirty;
    
    QGraphicsRectItem* back_clip_item;
    QGraphicsRectItem* front_clip_item;
    QGraphicsRectItem* graph_item;
    QGraphicsRectItem* graph_back_item;
    
    QMap<PlotItem*, PointSet> m_point_set;
    QMap<PlotItem*, PointHash> m_point_hash;
};

#endif // PLOT_H
