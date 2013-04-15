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

#ifndef POINT_H
#define POINT_H

#include <QtGui/QGraphicsObject>
#include <QtCore/QDebug>
#include <QtCore/QPropertyAnimation>

struct DataPoint
{
  double x;
  double y;
  
  operator QPointF() const;
};

Q_DECLARE_METATYPE(DataPoint)

QDebug& operator<<(QDebug& stream, const DataPoint& point);
bool operator==(const DataPoint& one, const DataPoint& other);

struct PointData
{
    PointData(int size, int symbol, const QColor& color, int state, bool transparent) : size(size), symbol(symbol), color(color), state(state), transparent(transparent) {}
    int size;
    int symbol;
    QColor color;
    int state;
    bool transparent;
};

class LabelItem : public QGraphicsTextItem
{
public:
	LabelItem(QGraphicsItem *parent = 0);
	LabelItem(const QString &text, QGraphicsItem *parent = 0);
	~LabelItem();

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
};

class Point : public QGraphicsObject
{
    Q_OBJECT
    Q_PROPERTY(QColor color READ color WRITE set_color)
    Q_PROPERTY(int symbol READ symbol WRITE set_symbol)
    Q_PROPERTY(int size READ size WRITE set_size)
    //Q_PROPERTY(QString label READ label WRITE set_label)
    Q_PROPERTY(DataPoint coordinates READ coordinates WRITE set_coordinates)
    Q_PROPERTY(bool marked READ is_marked WRITE set_marked STORED false)
    Q_PROPERTY(bool selected READ is_selected WRITE set_selected STORED false)
    
public:
    enum DisplayMode
    {
        DisplayPixmap,
        DisplayPath
    };
    
  /**
   * @brief Point symbol
   * 
   * The symbols list here matches the one from QwtPlotCurve. 
   **/
  enum Symbol {
    NoSymbol = -1,
    Ellipse = 0,
    Rect = 1,
    Diamond = 2,
    Triangle = 3,
    DTriangle = 4,
    UTriangle = 5,
    LTriangle = 6,
    RTriangle = 7,
    Cross = 8,
    XCross = 9,
    HLine = 10,
    VLine = 11,
    Star1 = 12,
    Star2 = 13,
    Hexagon = 14,
    UserStyle = 1000
  };
  
    enum StateFlag
    {
        Normal = 0x00,
        Marked = 0x01,
        Selected = 0x02
    };
    
    Q_DECLARE_FLAGS(State, StateFlag)
  
    enum 
    {
        Type = UserType + 1
    };
    
    virtual int type() const
    {
        return Type;
    }
    
    explicit Point(QGraphicsItem* parent = 0);
    Point(int symbol, QColor color, int size, QGraphicsItem* parent = 0);
    virtual ~Point();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    virtual QRectF boundingRect() const;
    
    void set_symbol(int symbol);
    int symbol() const;
    
    void set_color(const QColor& color);
    QColor color() const;
    
    void set_size(int size);
    int size() const;
    
    void set_display_mode(DisplayMode mode);
    DisplayMode display_mode() const;
    
    void set_state(State state);
    State state() const;
    void set_state_flag(StateFlag flag, bool on);
    bool state_flag(StateFlag flag) const;
    
    void set_selected(bool selected);
    bool is_selected() const;

    void set_marked(bool marked);
    bool is_marked() const;

    bool is_transparent();
    void set_transparent(bool transparent);
    
    DataPoint coordinates() const;
    void set_coordinates(const DataPoint& data_point);
    
    //void set_label(const QString& label);
    QString text() const;
    
    /**
    * Creates a path from a symbol and a size
    *
    * @param symbol the point symbol to use
    * @param size the size of the resulting path
    * @return a path that can be used in a QGraphicsPathItem or in QPainter::drawPath()
    **/
    static QPainterPath path_for_symbol(int symbol, int size);
    
    static QPixmap pixmap_for_symbol(int symbol, QColor color, int size);
    static QRectF rect_for_size(double size);
    
    static void clear_cache();

    static QHash<PointData, QPixmap> pixmap_cache;

    LabelItem* label;



private:
    static QPainterPath trianglePath(double d, double rot);
    static QPainterPath crossPath(double d, double rot);
    static QPainterPath hexPath(double d, bool star);

    int m_symbol;
    QColor m_color;
    int m_size;
    DisplayMode m_display_mode;
    State m_state;
    bool m_transparent;
    
    DataPoint m_coordinates;
    //QString m_label;
};

struct PointPosUpdater
{
  PointPosUpdater(QTransform t) : t(t) {}
  void operator()(Point* point)
  {
	QPointF p = t.map(QPointF(point->coordinates().x, point->coordinates().y));
    point->setPos(p);
    if (point->label)
    {
    	point->label->setPos(p);
    }
  }
  
private:
    QTransform t;
};

struct PointPropertyUpdater
{
    PointPropertyUpdater(const QByteArray& property, const QVariant& value) : property(property), value(value) {}
    void operator()(Point* point)
    {
        point->setProperty(property, value);
    }
    
private:
    QByteArray property;
    QVariant value;
};

#endif // POINT_H
