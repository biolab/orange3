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

#ifndef CURVE_H
#define CURVE_H

#include "plotitem.h"
#include "point.h"

#include <QtGui/QPen>
#include <QtGui/QBrush>
#include <QtCore/QtConcurrentMap>
#include <QtCore/QFutureWatcher>
#include <QtCore/QParallelAnimationGroup>
#include <QtCore/QtConcurrentRun>

struct PointPosMapper{
  PointPosMapper(const QTransform& t) : t(t) {}
  typedef QPointF result_type;
  result_type operator()(Point* p)
  {
    return t.map(p->coordinates());
  }
  
private:
    QTransform t;
};

struct PointUpdater
{
    PointUpdater(int symbol, QColor color, int size, Point::DisplayMode mode)
    {
        m_symbol = symbol;
        m_color = color;
        m_size = size;
        m_mode = mode;
    }
    
    void operator()(Point* point)
    {
        point->set_symbol(m_symbol);
        point->set_color(m_color);
        point->set_size(m_size);
        point->set_display_mode(m_mode);
    }
    
    private:
     int m_symbol;
     QColor m_color;
     int m_size;
     Point::DisplayMode m_mode;
     QTransform m_scale;
};

struct Updater
{
    Updater(double scale, const QPen& pen, const QBrush& brush, const QPainterPath& path)
    {
        m_scale = scale;
        m_pen = pen;
        m_brush = brush;
        m_path = path;
    }
    
    void operator()(QGraphicsPathItem* item)
    {
        item->setBrush(m_brush);
        item->setPen(m_pen);
        item->setScale(m_scale);
        item->setPath(m_path);
    }
    
    double m_scale;
    QPen m_pen;
    QBrush m_brush;
    QPainterPath m_path;
};
  
typedef QList< DataPoint > Data;

class Curve : public PlotItem
{
    Q_OBJECT
  
public:
  enum Style {
    Points = Qt::NoPen,
    Lines = Qt::SolidLine,
    Dots = Qt::DotLine,
    Sticks = 20,
    Steps,
    LinesPoints,
    UserCurve = 100
  };
  
  /**
   * @brief Default constructor
   * 
   * Constructs a Curve from a series of data points
   *
   * @param x_data A list of x coordinates of data points
   * @param y_data A list of y coordinates of data points
   * @param parent parent item
   **/
  Curve(const QList< double >& x_data, const QList< double >& y_data, QGraphicsItem* parent = 0);
  explicit Curve(QGraphicsItem* parent = 0);
  /**
   * Default destructor
   *
   **/
  virtual ~Curve();
    
  /**
   * @brief Update the curve
   * 
   * Moves all the points to their current locations, and changes their color, shape and size. 
   * Subclasses should reimplement this method to update their specific properties. 
   * 
   **/
   virtual void update_properties();
  
  Point* point_item(double x, double y, int size = 0, QGraphicsItem* parent = 0);
  
  QColor color() const;
  void set_color(const QColor& color);
  
  QPen pen() const;
  void set_pen(QPen pen);
  
  QBrush brush() const;
  void set_brush(QBrush brush);
  
  int point_size() const;
  void set_point_size(int size);
  
  int symbol() const;
  void set_symbol(int symbol);
  
  bool is_continuous() const;
  void set_continuous(bool continuous);

  Data data() const;
  void set_data(const QList<double> x_data, const QList<double> y_data);
  
  virtual QTransform graph_transform() const;
  virtual void set_graph_transform(const QTransform& transform);
  virtual void register_points();
  
  QRectF graphArea() const;
  void setGraphArea(const QRectF& area);
  
  int style() const;
  void set_style(int style);
  
  bool auto_update() const;
  void set_auto_update(bool auto_update);
  
  QTransform zoom_transform();
  virtual void set_zoom_transform(const QTransform& transform);
  
  QPainterPath continuous_path();

  enum UpdateFlag
  {
    UpdateNumberOfItems = 0x01,
    UpdatePosition = 0x02,
    UpdateSymbol = 0x04,
    UpdateSize = 0x08,
    UpdatePen = 0x10,
    UpdateBrush = 0x20,
    UpdateContinuous = 0x40,
    UpdateZoom = 0x80,
    UpdateAll = 0xFF
  };
  
  Q_DECLARE_FLAGS(UpdateFlags, UpdateFlag)
  
  void set_dirty(UpdateFlags flags = UpdateAll);
  
  template <class Sequence, class Updater>
  void update_items(const Sequence& sequence, Updater updater, Curve::UpdateFlag flag);
    
  template <class T>
  void update_point_properties(const QByteArray& property, const QList< T >& values, bool animate = true);

  template <class T>
  void update_point_properties_threaded(const QByteArray& property, const QList<T>& values);
  
  void update_point_properties_same(const QByteArray& property, const QVariant& value, bool animate);
  
  template <class T>
  void resize_item_list(QList< T* >& list, int size);

  void set_points(const QList<Point*>& points);
  QList<Point*> points();
  
  bool labels_on_marked();
  void set_labels_on_marked(bool value);

  QMap<UpdateFlag, QFuture<void> > m_currentUpdate;

protected:
  Curve::UpdateFlags needs_update();
  void set_updated(Curve::UpdateFlags flags);
  
  void cancel_all_updates();
  void update_number_of_items();
  
  void checkForUpdate();
  void changeContinuous();
  
  bool use_animations();
  
public slots:
    void update_point_coordinates();
    void update_point_positions();
  
private slots:
    void pointMapFinished();

private:
  QColor m_color;
  int m_pointSize;
  int m_symbol;
  int m_style;
  bool m_continuous;
  Data m_data;
  QTransform m_graphTransform;
  QList<Point*> m_pointItems;
  UpdateFlags m_needsUpdate;
  bool m_autoUpdate;
  QGraphicsPathItem* m_lineItem;
  bool m_labels_on_marked;

  QPen m_pen;
  QBrush m_brush;
  QTransform m_zoom_transform;
  QMap<QByteArray, QFuture<void> > m_property_updates;
  QFutureWatcher<QPointF> m_pos_watcher;
  QFutureWatcher<void> m_coords_watcher;
  
};

template <class Sequence, class Updater>
void Curve::update_items(const Sequence& sequence, Updater updater, Curve::UpdateFlag flag)
{
    if (m_currentUpdate.contains(flag) && m_currentUpdate[flag].isRunning())
    {
        m_currentUpdate[flag].cancel();
        m_currentUpdate[flag].waitForFinished();
    }
    if (!sequence.isEmpty())
    {
        m_currentUpdate[flag] = QtConcurrent::map(sequence, updater);
    }
}

template < class T >
void Curve::update_point_properties(const QByteArray& property, const QList< T >& values, bool animate)
{
    if (m_property_updates.contains(property))
    {
        m_property_updates[property].cancel();
        m_property_updates[property].waitForFinished();
    }
    
    update_number_of_items();
    
    int n = m_pointItems.size();
    if (n != values.size())
    {
        if (values.isEmpty())
        {
            update_point_properties_same(property, T(), animate);
        }
        else
        {
            update_point_properties_same(property, values.first(), animate);
        }
        
        return;
    }
    
    if (animate && use_animations())
    {
        QParallelAnimationGroup* group = new QParallelAnimationGroup(this);
        for (int i = 0; i < n; ++i)
        {
            QPropertyAnimation* a = new QPropertyAnimation(m_pointItems[i], property, m_pointItems[i]);
            a->setEndValue(values[i]);
            group->addAnimation(a);
        }
        group->start(QAbstractAnimation::DeleteWhenStopped);
    }
    else
    {
        m_property_updates[property] = QtConcurrent::run(this, &Curve::update_point_properties_threaded<T>, property, values);
    }
}

template < class T >
void Curve::update_point_properties_threaded(const QByteArray& property, const QList< T >& values)
{
    const int n = values.size();
    if (n != m_pointItems.size())
    {
	return;
    }
    for (int i = 0; i < n; ++i)
    {
        m_pointItems[i]->setProperty(property, QVariant::fromValue<T>(values[i]));
    }
}

template <class T>
void Curve::resize_item_list(QList< T* >& list, int size)
{
    int n = list.size();  
    if (n > size)
  {
    qDeleteAll(list.constBegin() + size, list.constEnd());
    list.erase(list.begin() + size, list.end());
  }
  else if (n < size)
  {  
#if QT_VERSION >= 0x040700
    list.reserve(size);
#endif
    for (int i = 0; i < (size-n); ++i)
    {
      list << new T(this);
    }
  }
}


Q_DECLARE_OPERATORS_FOR_FLAGS(Curve::UpdateFlags)


#endif // CURVE_H
