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

#include "curve.h"

#include <QtCore/QDebug>
#include <QtGui/QBrush>
#include <QtGui/QPen>

#include <QtCore/qmath.h>
#include <QtCore/QtConcurrentRun>
#include <QtCore/QFutureWatcher>

#include "point.h"
#include "plot.h"

#include <QtCore/QParallelAnimationGroup>
#include <QtCore/QCoreApplication>

Curve::Curve(const QList< double >& x_data, const QList< double >& y_data, QGraphicsItem* parent): PlotItem(parent)
{
    // Don't make any calls to update_properties() until the constructor is finished
    // Otherwise, the program hangs if this is called from a subclass constructor
    m_autoUpdate = false; 
    
    m_style = Points;
    m_continuous = false;
    m_needsUpdate = UpdateAll;
    m_lineItem = new QGraphicsPathItem(this);
    set_data(x_data, y_data);
    QObject::connect(&m_pos_watcher, SIGNAL(finished()), SLOT(pointMapFinished()));
    QObject::connect(&m_coords_watcher, SIGNAL(finished()), SLOT(update_point_positions()));
    m_autoUpdate = true;
}

Curve::Curve(QGraphicsItem* parent): PlotItem(parent)
{
    m_continuous = false;
    m_autoUpdate = true;
    m_style = Points;
    m_lineItem = new QGraphicsPathItem(this);
    m_needsUpdate = 0;
    QObject::connect(&m_pos_watcher, SIGNAL(finished()), SLOT(pointMapFinished()));
    QObject::connect(&m_coords_watcher, SIGNAL(finished()), SLOT(update_point_positions()));
}


Curve::~Curve()
{
    cancel_all_updates();
}

void Curve::update_number_of_items()
{
  cancel_all_updates();
  if (m_continuous || (m_data.size() == m_pointItems.size()))
  {
    m_needsUpdate &= ~UpdateNumberOfItems;
    return;
  }
  if (m_pointItems.size() != m_data.size())
  {
    resize_item_list<Point>(m_pointItems, m_data.size());
    register_points();
  }
  Q_ASSERT(m_pointItems.size() == m_data.size());
}

void Curve::update_properties()
{
  cancel_all_updates();
  
  bool lines = false;
  bool points = false;
  
  switch (m_style)
  {
      case Points:
          points = true;
          break;
          
      case Lines:
      case Dots:
          lines = true;
          break;
          
      case LinesPoints:
          lines = true;
          points = true;
          break;
          
      default:
          lines = m_continuous;
          points = !m_continuous;
          break;
  }
  
  m_lineItem->setVisible(lines);
  
  if (lines)
  {
    QPen p = m_pen;
    p.setCosmetic(true);
    p.setStyle((Qt::PenStyle)m_style);
    m_lineItem->setPen(p);
    m_lineItem->setPath(continuous_path());
  } 
  
  if (points)
  {
    
    if (m_pointItems.size() != m_data.size())
    {
        update_number_of_items();
    }
    
    // Move, resize, reshape and/or recolor the items
    if (m_needsUpdate & UpdatePosition)
    {
        update_point_coordinates();
    } 
    
    if (m_needsUpdate & (UpdateBrush | UpdatePen | UpdateSize | UpdateSymbol) )
    {
        update_items(m_pointItems, PointUpdater(m_symbol, m_color, m_pointSize, Point::DisplayPath), UpdateSymbol);
    }
    m_needsUpdate = 0;
  }
  else
  {
      qDeleteAll(m_pointItems);
      m_pointItems.clear();
  }
}

Point* Curve::point_item(double x, double y, int size, QGraphicsItem* parent)
{
  if (size == 0)
  {
    size = point_size();
  }
  if (parent == 0)
  {
    parent = this;
  }
  Point* item = new Point(m_symbol, m_color, m_pointSize, parent);
  item->setPos(x,y);
  return item;
}

Data Curve::data() const
{
  return m_data;
}

void Curve::set_data(const QList< double > x_data, const QList< double > y_data)
{
  Q_ASSERT(x_data.size() == y_data.size());
  int n = qMin(x_data.size(), y_data.size());
  qDebug() << "Curve::set_data with" << n << "points";
  if (n != m_data.size())
  {
    m_needsUpdate |= UpdateNumberOfItems;
  }
  m_data.clear();

#if QT_VERSION >= 0x040700
  m_data.reserve(n);
#endif

  for (int i = 0; i < n; ++i)
  {
    DataPoint p;
    p.x = x_data[i];
    p.y = y_data[i];
    m_data.append(p);
  }
  set_data_rect(rect_from_data(x_data, y_data));
  m_needsUpdate |= UpdatePosition;
  checkForUpdate();
}

QTransform Curve::graph_transform() const
{
  return m_graphTransform;
}

void Curve::set_graph_transform(const QTransform& transform)
{
  if (transform == m_graphTransform)
  {
    return;
  }
  m_needsUpdate |= UpdatePosition;
  m_graphTransform = transform;
  checkForUpdate();
}

bool Curve::is_continuous() const
{
  return m_continuous;
}

void Curve::set_continuous(bool continuous)
{
  if (continuous == m_continuous)
  {
    return;
  }
  m_continuous = continuous;
  m_needsUpdate |= UpdateContinuous;
  checkForUpdate();
}

QColor Curve::color() const
{
  return m_color;
}

void Curve::set_color(const QColor& color)
{
    m_color = color;
    set_pen(color);
    set_brush(color);
}

QPen Curve::pen() const
{
    return m_pen;
}

void Curve::set_pen(QPen pen)
{
    m_pen = pen;
    m_needsUpdate |= UpdatePen;
    checkForUpdate();
}

QBrush Curve::brush() const
{
    return m_brush;
}

void Curve::set_brush(QBrush brush)
{
    m_brush = brush;
    m_needsUpdate |= UpdateBrush;
    checkForUpdate();
}

int Curve::point_size() const
{
  return m_pointSize;
}

void Curve::set_point_size(int size)
{
  if (size == m_pointSize)
  {
    return;
  }
  
  m_pointSize = size;
  m_needsUpdate |= UpdateSize;
  checkForUpdate();
}

int Curve::symbol() const
{
  return m_symbol;
}

void Curve::set_symbol(int symbol)
{
  if (symbol == m_symbol)
  {
    return;
  }
  m_symbol = symbol;
  m_needsUpdate |= UpdateSymbol;
  checkForUpdate();
}

int Curve::style() const
{
    return m_style;
}

void Curve::set_style(int style)
{
    m_style = style;
    m_needsUpdate |= UpdateAll;
    checkForUpdate();
}



bool Curve::auto_update() const
{
  return m_autoUpdate;
}

void Curve::set_auto_update(bool auto_update)
{
  m_autoUpdate = auto_update;
  checkForUpdate();
}

void Curve::checkForUpdate()
{
  if ( m_autoUpdate && m_needsUpdate )
  {
    update_properties();
  }
}

void Curve::changeContinuous()
{
  cancel_all_updates();
  if (m_continuous)
  {
    qDeleteAll(m_pointItems);
    m_pointItems.clear();
    
    if (!m_lineItem)
    {
      m_lineItem = new QGraphicsPathItem(this);
    }
  } else {
    delete m_lineItem;
    m_lineItem = 0;
  }
  register_points();
}

void Curve::set_dirty(Curve::UpdateFlags flags)
{
    m_needsUpdate |= flags;
    checkForUpdate();
}

void Curve::set_zoom_transform(const QTransform& transform)
{
    m_zoom_transform = transform;
    m_needsUpdate |= UpdateZoom;
    checkForUpdate();
}

QTransform Curve::zoom_transform()
{
    return m_zoom_transform;
}

void Curve::cancel_all_updates()
{
    QMap<UpdateFlag, QFuture< void > >::iterator it = m_currentUpdate.begin();
    QMap<UpdateFlag, QFuture< void > >::iterator end = m_currentUpdate.end();
    for (it; it != end; ++it)
    {
        if (it.value().isRunning())
        {
            it.value().cancel();
        }
    }
    for (it = m_currentUpdate.begin(); it != end; ++it)
    {
        if (it.value().isRunning())
        {
            it.value().waitForFinished();
        }
    }
    m_currentUpdate.clear();
    
    m_coords_watcher.blockSignals(true);
    m_coords_watcher.cancel();
    m_coords_watcher.waitForFinished();
    m_coords_watcher.blockSignals(false);
    
    m_pos_watcher.blockSignals(true);
    m_pos_watcher.cancel();
    m_pos_watcher.waitForFinished();
    m_pos_watcher.blockSignals(false);
}

void Curve::register_points()
{
    Plot* p = plot();
    if (p)
    {
        p->remove_all_points(this);
        p->add_points(m_pointItems, this);
    }
}

Curve::UpdateFlags Curve::needs_update()
{
    return m_needsUpdate;
}

void Curve::set_updated(Curve::UpdateFlags flags)
{
    m_needsUpdate &= ~flags;
}

void Curve::set_points(const QList< Point* >& points)
{
    if (points == m_pointItems)
    {
        return;
    }
    m_pointItems = points;
    register_points();
}

QList< Point* > Curve::points()
{
    return m_pointItems;
}

void Curve::set_labels_on_marked(bool value)
{
	m_labels_on_marked = value;
}

bool Curve::labels_on_marked()
{
	return m_labels_on_marked;
}

void Curve::update_point_coordinates()
{
    if (m_coords_watcher.isRunning())
    {
        m_coords_watcher.blockSignals(true);
        m_coords_watcher.cancel();
        m_coords_watcher.waitForFinished();
        m_coords_watcher.blockSignals(false);
    }
    m_coords_watcher.setFuture(QtConcurrent::run(this, &Curve::update_point_properties_threaded<DataPoint>, QByteArray("coordinates"), m_data));
}

void Curve::update_point_positions()
{
    if (m_pos_watcher.isRunning())
    {
        m_pos_watcher.blockSignals(true);
        m_pos_watcher.cancel();
        m_pos_watcher.waitForFinished();
        m_pos_watcher.blockSignals(false);
    }
    if (m_pointItems.isEmpty())
    {
        return;
    }
    if (use_animations())
    {
        m_pos_watcher.setFuture(QtConcurrent::mapped(m_pointItems, PointPosMapper(m_graphTransform)));
    }
    else
    {
        update_items(m_pointItems, PointPosUpdater(m_graphTransform), UpdatePosition);
    }
}

void Curve::pointMapFinished()
{
    if (m_pointItems.size() != m_pos_watcher.future().results().size())
    {
        // The calculation that just finished is already out of date, ignore it
        return;
    }
    QParallelAnimationGroup* group = new QParallelAnimationGroup(this);
    int n = m_pointItems.size();
    for (int i = 0; i < n; ++i)
    {
        /* 
         * If a point was just created, its position is (0,0)
         * In this case, animating it would create more confusion that good
         * So we just move it without an animation. 
         * This is the case (for example) for the anchor curve in RadViz
         */
        if (m_pointItems[i]->pos().isNull())
        {
            m_pointItems[i]->setPos(m_pos_watcher.resultAt(i));
            // move point label
            if (m_pointItems[i]->label)
			{
            	m_pointItems[i]->label->setPos(m_pos_watcher.resultAt(i));
			}
        }
        else
        {
            QPropertyAnimation* a = new QPropertyAnimation(m_pointItems[i], "pos", m_pointItems[i]);
            a->setEndValue(m_pos_watcher.resultAt(i));
            group->addAnimation(a);

            // move point label
            if (m_pointItems[i]->label)
            {
				QPropertyAnimation* b = new QPropertyAnimation(m_pointItems[i]->label, "pos", m_pointItems[i]->label);
				b->setEndValue(m_pos_watcher.resultAt(i));
				group->addAnimation(b);
            }
        }
    }
    group->start(QAbstractAnimation::DeleteWhenStopped);
}

bool Curve::use_animations()
{
    return plot() && plot()->animate_points;
}

void Curve::update_point_properties_same(const QByteArray& property, const QVariant& value, bool animate) {
    int n = m_pointItems.size();

    if (animate && use_animations())
    {
        QParallelAnimationGroup* group = new QParallelAnimationGroup(this);
        for (int i = 0; i < n; ++i)
        {
            QPropertyAnimation* a = new QPropertyAnimation(m_pointItems[i], property, m_pointItems[i]);
            a->setEndValue(value);
            group->addAnimation(a);
        }
        group->start(QAbstractAnimation::DeleteWhenStopped);
    }
    else
    {
        m_property_updates[property] = QtConcurrent::map(m_pointItems, PointPropertyUpdater(property, value));
    }
}

QPainterPath Curve::continuous_path()
{
    QPainterPath path;
    if (m_data.isEmpty())
    {
        return path;
    }
    path.moveTo(m_data[0]);
    int n = m_data.size();
    QPointF p;
    for (int i = 1; i < n; ++i)
    {
        path.lineTo(m_data[i]);
    }
    return m_graphTransform.map(path);
}

#include "curve.moc"

