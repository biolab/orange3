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

#include "plot.h"
#include "plotitem.h"
#include "point.h"

#include <QtCore/QDebug>
#include <QtCore/qmath.h>
#include <limits>

inline uint qHash(const DataPoint& pos)
{
    return pos.x + pos.y;
}

inline double distance(const QPointF& one, const QPointF& other)
{
    // For speed, we use the slightly wrong method, also known as Manhattan distance
    return (one - other).manhattanLength();
}

template <class Area>
void set_points_state(Area area, QGraphicsScene* scene, Point::StateFlag flag, Plot::SelectionBehavior behavior)
{
    /*
     * NOTE: I think it's faster to rely on Qt to get all items in the current rect
     * than to iterate over all points on the graph and check which of them are 
     * inside the specified rect
     */
    foreach (QGraphicsItem* item, scene->items(area, Qt::IntersectsItemBoundingRect))
    {
        Point* point = dynamic_cast<Point*>(item);
        if (point)
        {
            point->set_state_flag(flag, behavior == Plot::AddSelection || (behavior == Plot::ToggleSelection && !point->state_flag(flag)));
        }
    }
}

Plot::Plot(QWidget* parent):
QGraphicsView(parent)
{
    setScene(new QGraphicsScene(this));
    
    back_clip_item = new QGraphicsRectItem();
    back_clip_item->setPen(Qt::NoPen);
    back_clip_item->setFlag(QGraphicsItem::ItemClipsChildrenToShape, true);
    back_clip_item->setZValue(-1000);
    scene()->addItem(back_clip_item);
    
    front_clip_item = new QGraphicsRectItem();
    front_clip_item->setPen(Qt::NoPen);
    front_clip_item->setFlag(QGraphicsItem::ItemClipsChildrenToShape, true);
    front_clip_item->setZValue(0);
    scene()->addItem(front_clip_item);
    
    graph_back_item = new QGraphicsRectItem(back_clip_item);
    graph_back_item->setPen(Qt::NoPen);
    graph_item = new QGraphicsRectItem(front_clip_item);
    graph_item->setPen(Qt::NoPen);
}

Plot::~Plot()
{
    foreach (PlotItem* item, m_items)
    {
        remove_item(item);
    }
}

void Plot::add_item(PlotItem* item)
{
    if (m_items.contains(item))
    {
        qWarning() << "Item is already in this graph";
        return;
    }
    item->m_plot = this;
    item->setParentItem(item->is_in_background() ? graph_back_item : graph_item);
    m_items << item;
    item->register_points();
    item->update_properties();
}

void Plot::remove_item(PlotItem* item)
{
    if (m_items.contains(item))
    {
        item->setParentItem(0);
        m_items.removeAll(item);
        item->m_plot = 0;
        if (scene()->items().contains(item))
        {
            scene()->removeItem(item);
        }
    }
    else
    {
        qWarning() << "Trying to remove an item that doesn't belong to this graph";
    }
    remove_all_points(item);
}

void Plot::set_item_in_background(PlotItem* item, bool bg)
{
    item->setParentItem(bg ? graph_back_item : graph_item);
}

QList< PlotItem* > Plot::plot_items()
{
    return m_items;
}

QRectF Plot::data_rect_for_axes(int x_axis, int y_axis)
{
    QRectF r;
    QPair<int,int> axes = qMakePair(x_axis, y_axis);
    foreach (PlotItem* item, m_items)
    {
        if (item->is_auto_scale() && item->axes() == axes)
        {
            r |= item->data_rect();
        }
    }
    return r;
}

QPair< double, double > Plot::bounds_for_axis(int axis)
{
    QRectF y_r;
    QRectF x_r;
    foreach (PlotItem* item, m_items)
    {
        if (item->is_auto_scale())
        {
            if (item->axes().first == axis)
            {
               x_r |= item->data_rect(); 
            }
            else if (item->axes().second == axis)
            {
                y_r |= item->data_rect();
            }
        }
    }
    if (x_r.isValid())
    {
        return qMakePair(x_r.left(), x_r.right());
    }
    else if (y_r.isValid())
    {
        return qMakePair(y_r.top(), y_r.bottom());
    }
    return qMakePair(0.0, 0.0);
}

void Plot::set_dirty() 
{
    m_dirty = true;
}

void Plot::set_clean() 
{
    m_dirty = false;
}

bool Plot::is_dirty() 
{
    return m_dirty;
}

void Plot::set_graph_rect(const QRectF rect) 
{
    foreach (QGraphicsRectItem* item, QList<QGraphicsRectItem*>() << front_clip_item << back_clip_item << graph_item << graph_back_item)
    {
        item->setRect(rect);
    }
}

void Plot::set_zoom_transform(const QTransform& zoom)
{
    graph_item->setTransform(zoom);
    graph_back_item->setTransform(zoom);
    foreach (PlotItem* item, plot_items())
    {
        item->set_zoom_transform(zoom);
    }
}

void Plot::mark_points(const QRectF& rect, Plot::SelectionBehavior behavior)
{
    if (behavior == ReplaceSelection)
    {
        bool b = blockSignals(true);
        unmark_all_points();
        behavior = AddSelection;
        blockSignals(b);
    }
    set_points_state(rect, scene(), Point::Marked, behavior);
    emit marked_points_changed();
}

void Plot::mark_points(const QPolygonF& area, Plot::SelectionBehavior behavior)
{
    if (behavior == ReplaceSelection)
    {
        bool b = blockSignals(true);
        unmark_all_points();
        behavior = AddSelection;
        blockSignals(b);
    }
    set_points_state(area, scene(), Point::Marked, behavior);
    emit marked_points_changed();
}

void Plot::select_points(const QRectF& rect, Plot::SelectionBehavior behavior)
{
    if (behavior == ReplaceSelection)
    {
        bool b = blockSignals(true);
        unselect_all_points();
        behavior = AddSelection;
        blockSignals(b);
    }
    set_points_state(rect, scene(), Point::Selected, behavior);
    emit selection_changed();
}

void Plot::select_points(const QPolygonF& area, Plot::SelectionBehavior behavior)
{
    if (behavior == ReplaceSelection)
    {
        bool b = blockSignals(true);
        unselect_all_points();
        behavior = AddSelection;
        blockSignals(b);
    }
    set_points_state(area, scene(), Point::Selected, behavior);
    emit selection_changed();
}

QList< bool > Plot::selected_points(const QList< double > x_data, const QList< double > y_data)
{
    Q_ASSERT(x_data.size() == y_data.size());
    const int n = qMin(x_data.size(), y_data.size());
    QList<bool> selected;
#if QT_VERSION >= 0x040700
    selected.reserve(n);
#endif
    DataPoint p;
    for (int i = 0; i < n; ++i)
    {
        p.x = x_data[i];
        p.y = y_data[i];
        selected << (selected_point_at(p) ? true : false);
    }
    return selected;
}

QList< Point* > Plot::selected_points()
{
    QList<Point*> list;
    foreach (Point* p, all_points())
    {
        if (p->is_selected())
        {
            list << p;
        }
    }
    qDebug() << "Found" << list.size() << "selected points";
    return list;
}

QList< Point* > Plot::marked_points()
{
    QList<Point*> list;
    foreach (Point* point, all_points())
    {
        if (point->is_marked())
        {
            list.append(point);
        }
    }
    return list;
}

Point* Plot::selected_point_at(const DataPoint& pos)
{
    foreach (PlotItem* item, plot_items())
    {
        if (m_point_set.contains(item) && m_point_set[item].contains(pos))
        {
            foreach (Point* p, m_point_hash[item].values(pos))
            {
                if (p->is_selected())
                {
                    return p;
                }
            }
        }
    }
    return 0;
}

Point* Plot::point_at(const DataPoint& pos)
{
    foreach (const PointHash& hash, m_point_hash)
    {
        if (hash.contains(pos))
        {
            return hash.values(pos).first();
        }
    }
    return 0;
}

Point* Plot::nearest_point(const QPointF& pos)
{
    QPointF zoomedPos = graph_item->transform().inverted().map(pos);
    QPair<double, Point*> closest_point;
    closest_point.first = std::numeric_limits<double>::max();
    closest_point.second = 0;
    
    foreach (const PointHash hash, m_point_hash)
    {
        foreach (Point* p, hash)
        {
            const double d = distance(p->pos(), zoomedPos);
            if (d < closest_point.first)
            {
                closest_point.first = d;
                closest_point.second = p;
            }
        }
    }
    
    
    if (closest_point.second)
    {
        // In case of zooming, we want the actual distance on the screen, 
        // rather then the distance on the non-zoomed canvas
        closest_point.first = distance(graph_item->transform().map(closest_point.second->pos()), pos);
    }
		
    if(closest_point.second && closest_point.first <= closest_point.second->size())
    {
        return closest_point.second;
    }
    else
    {
        return 0;
    }
}

void Plot::add_point(Point* point, PlotItem* parent)
{
    const DataPoint pos = point->coordinates();
    m_point_set[parent].insert(pos);
    m_point_hash[parent].insert(pos, point);
}

void Plot::add_points(const QList< Point* >& items, PlotItem* parent)
{
    foreach (Point* p, items)
    {
        add_point(p, parent);
    }
}

void Plot::remove_point(Point* point, PlotItem* parent)
{
    const DataPoint pos = point->coordinates();
    if (m_point_set.contains(parent) && m_point_set[parent].contains(pos))
    {
        m_point_set[parent].remove(pos);
        m_point_hash[parent].remove(pos, point);
    }
}

void Plot::remove_all_points(PlotItem* parent)
{
    m_point_set.remove(parent);
    m_point_hash.remove(parent);
}

void Plot::unmark_all_points()
{
    foreach (Point* point, all_points())
    {
        point->set_marked(false);
    }
    emit marked_points_changed();
}

void Plot::unselect_all_points()
{
    foreach (Point* point, all_points())
    {
        point->set_selected(false);
    }
    emit selection_changed();
}

void Plot::selected_to_marked()
{
	foreach (const PointHash& hash, m_point_hash)
	{
		foreach (Point* point, hash)
		{
			point->set_marked(point->is_selected());
			point->set_selected(false);
		}
	}
	emit selection_changed();
    emit marked_points_changed();
}

void Plot::marked_to_selected()
{
	foreach (const PointHash& hash, m_point_hash)
	{
		foreach (Point* point, hash)
		{
			point->set_selected(point->is_marked());
			point->set_marked(false);
		}
	}
	emit selection_changed();
    emit marked_points_changed();
}

void Plot::mark_points(const Data& data, Plot::SelectionBehavior behavior)
{
    foreach (const PointHash& hash, m_point_hash)
    {
        foreach (Point* point, hash)
        {
            if (data.contains(point->coordinates()))
            {
                qDebug() << "Found a point, marking it";
                point->set_marked(behavior == AddSelection || behavior == ReplaceSelection || (behavior == ToggleSelection && !point->is_marked()));
            }
            else if (behavior == ReplaceSelection)
            {
                point->set_marked(false);
            }
        }
    }
}

void Plot::select_points(const Data& data, Plot::SelectionBehavior behavior)
{
    foreach (const PointHash& hash, m_point_hash)
    {
        foreach (Point* point, hash)
        {
            if (data.contains(point->coordinates()))
            {
                point->set_selected(behavior == AddSelection || (behavior == ToggleSelection && !point->is_selected()));
            }
            else if (behavior == ReplaceSelection)
            {
                point->set_selected(false);
            }
        }
    }
}

void Plot::move_selected_points(const DataPoint& d)
{
	foreach (Point* p, all_points())
	{
		if (p->is_selected())
		{
			DataPoint c = p->coordinates();
			c.x += d.x;
			c.y += d.y;
			p->set_coordinates(c);
		}
	}
}

void Plot::emit_marked_points_changed()
{
	emit marked_points_changed();
}

void Plot::emit_selection_changed()
{
	emit selection_changed();
}

QList< Point* > Plot::all_points()
{
    QList<Point*> list;
    foreach (PlotItem* item, plot_items())
    {
        Curve* curve = qobject_cast<Curve*>(item);
        if (!curve)
        {
            continue;
        }
        list << curve->points();
    }
    return list;
}

#include "plot.moc"
