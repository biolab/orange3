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

#include "unconnectedlinescurve.h"
#include <QtGui/QPen>
#include <QtCore/QDebug>

struct PointMapper
{
    typedef QPointF result_type;
    PointMapper(const QTransform& t) : t(t) {}
    
    QPointF operator()(const DataPoint& p)
    {
        return t.map(p);
    }
    
    QTransform t;
};

struct PathReducer
{
    PathReducer() : closed(true) {}
    void operator()(QPainterPath& path, const QPointF point)
    {
        if (closed)
        {
            path.moveTo(point);
        }
        else
        {
            path.lineTo(point);
        }
        closed = !closed;
    }
    
    bool closed;
};

UnconnectedLinesCurve::UnconnectedLinesCurve(QGraphicsItem* parent): Curve(parent)
{
    m_path_item = new QGraphicsPathItem(this);
    m_path_watcher = new QFutureWatcher<QPainterPath>(this);
    connect(m_path_watcher, SIGNAL(finished()), SLOT(path_calculated()));
    setFlag(ItemHasNoContents);
}

UnconnectedLinesCurve::~UnconnectedLinesCurve()
{

}

void UnconnectedLinesCurve::update_properties()
{
    cancel_all_updates();
    if (needs_update() & UpdatePosition)
    {
        m_path_watcher->setFuture(QtConcurrent::mappedReduced<QPainterPath>(data(), PointMapper(graph_transform()), PathReducer(), QtConcurrent::OrderedReduce | QtConcurrent::SequentialReduce));
    }
    if (needs_update() & UpdatePen)
    {   
        QPen p = pen();
        p.setCosmetic(true);
        m_path_item->setPen(p);
    }
    set_updated(Curve::UpdateAll);
}

void UnconnectedLinesCurve::path_calculated()
{
    m_path_item->setPath(m_path_watcher->result());
}

#include "unconnectedlinescurve.moc"