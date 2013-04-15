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

#include "point.h"
#include "curve.h"

#include <QtGui/QPainter>
#include <QtCore/QDebug>
#include <QtCore/qmath.h>
#include <QtGui/QStyleOptionGraphicsItem>

QHash<PointData, QPixmap> Point::pixmap_cache;

uint qHash(const PointData& data)
{
    // uint has 32 bits:
    uint ret = data.size;
    // size only goes up to 20, so 5 bits is enough
    ret |= data.symbol << 5;
    // symbol is less than 16, so 4 bits will do
    ret |= data.state << 9;
    // state is currently only two bits
    ret |= data.transparent << 11;
    // QRgb takes the full uins, so we just XOR by it
    ret ^= data.color.rgba();
    return ret;
}

bool operator==(const PointData& one, const PointData& other)
{
    return one.symbol == other.symbol && one.size == other.size && one.state == other.state && one.color == other.color;
}

QDebug& operator<<(QDebug& stream, const DataPoint& point)
{
	return stream.maybeSpace() << "DataPoint(" << point.x << ','<< point.y << ')';
}

bool operator==(const DataPoint& one, const DataPoint& other)
{
    return one.x == other.x && one.y == other.y;
}

DataPoint::operator QPointF() const
{
    return QPointF(x, y);
}

LabelItem::LabelItem(QGraphicsItem* parent): QGraphicsTextItem(parent)
{
}

LabelItem::LabelItem(const QString &text, QGraphicsItem *parent): QGraphicsTextItem(text, parent)
{
}

LabelItem::~LabelItem()
{
}

void LabelItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
	/*
    Q_UNUSED(widget);
    if (dd->control) {
        painter->save();
        QRectF r = option->exposedRect;
        painter->translate(-dd->controlOffset());
        r.translate(dd->controlOffset());

        QTextDocument *doc = dd->control->document();
        QTextDocumentLayout *layout = qobject_cast<QTextDocumentLayout *>(doc->documentLayout());

        // the layout might need to expand the root frame to
        // the viewport if NoWrap is set
        if (layout)
            layout->setViewport(dd->boundingRect);

        dd->control->drawContents(painter, r);

        if (layout)
            layout->setViewport(QRect());

        painter->restore();
    }

    if (option->state & (QStyle::State_Selected | QStyle::State_HasFocus))
        qt_graphicsItem_highlightSelected(this, painter, option);
    */
	QGraphicsTextItem::paint(painter, option, widget);
}

Point::Point(int symbol, QColor color, int size, QGraphicsItem* parent): QGraphicsObject(parent),
 m_symbol(symbol),
 m_color(color),
 m_size(size)
{
    m_display_mode = DisplayPath;
    m_transparent = true;
    setFlag(ItemIgnoresTransformations);
    label = NULL;
}

Point::Point(QGraphicsItem* parent): QGraphicsObject(parent)
{
    m_symbol = Ellipse;
    m_color = Qt::black;
    m_size = 5;
    m_display_mode = DisplayPath;
    m_transparent = true;
    setFlag(ItemIgnoresTransformations);
    label = NULL;
}
  
void Point::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)
    
    const PointData key(m_size, m_symbol, m_color, m_state, m_transparent);
    // We make the pixmap slighly larger because the point outline has non-zero width
    const int ps = m_size + 4;
    if (!pixmap_cache.contains(key))
    {
        if (m_display_mode == DisplayPath)
        {
            QBrush brush(m_color);
            QPixmap pixmap(ps, ps);
            pixmap.fill(Qt::transparent);
            QPainter p;
            
            QPen pen(m_color);
            pen.setWidth(qMin(2, m_size/6));
            
            p.begin(&pixmap);
            p.setRenderHints(painter->renderHints() | QPainter::Antialiasing);
            if (m_state & Selected)
            {
                brush.setColor(m_color);
                //pen.setStyle(Qt::NoPen);
            }
            else if (m_state & Marked)
            {
                /*
            	QRadialGradient g(0.5*ps, 0.5*ps, 0.5*m_size);
                g.setColorAt(0, m_color);
                g.setColorAt(0.5, Qt::transparent);
                g.setColorAt(1, m_color);
                brush = QBrush(g);
                pen.setStyle(Qt::NoPen);
                */
            	//QColor c = Qt::darkGray;
            	//c.setAlpha(150);
            	QColor c = brush.color();
				//c.setAlpha(m_color.alpha()/6);
				brush.setColor(c);
				pen.setColor(Qt::black);
				pen.setWidth(qMin(3, m_size/3));

            }
            else
            {
                QColor c = brush.color();
                c.setAlpha(m_color.alpha()/6);
                brush.setColor(c);
            }
            const QPainterPath path = path_for_symbol(m_symbol, m_size).translated(0.5*ps, 0.5*ps);

            if (!m_transparent)
            {
            	p.setBrush(Qt::white);
            	p.drawPath(path);
            }

            p.setBrush(brush);
            p.setPen(pen);
            p.drawPath(path);
            pixmap_cache.insert(key, pixmap);
        } 
        else if (m_display_mode == DisplayPixmap)
        {
            pixmap_cache.insert(key, pixmap_for_symbol(m_symbol, m_color, m_size));
        }
    }
    painter->drawPixmap(QPointF(-0.5*ps, -0.5*ps), pixmap_cache.value(key));
    /*
    if (!m_label.isEmpty())
    {        
        QFontMetrics metrics = option->fontMetrics;
        int th = metrics.height();
        int tw = metrics.width(m_label);
        QRect r(-tw/2, 0, tw, th);
        //painter->fillRect(r, QBrush(Qt::white));
        painter->drawText(r, Qt::AlignHCenter, m_label);
    }
    */
}

QRectF Point::boundingRect() const
{
    return rect_for_size(m_size);
}

Point::~Point()
{
}

QColor Point::color() const
{
    return m_color;
}

void Point::set_color(const QColor& color)
{
    m_color = color;
    update();
}

Point::DisplayMode Point::display_mode() const
{
    return m_display_mode;
}

void Point::set_display_mode(Point::DisplayMode mode)
{
    m_display_mode = mode;
    update();
}

int Point::size() const
{
    return m_size;
}

void Point::set_size(int size)
{
    m_size = size;
    update();
}

int Point::symbol() const
{
    return m_symbol;
}

void Point::set_symbol(int symbol)
{
    m_symbol = symbol;
    update();
}

QPainterPath Point::path_for_symbol(int symbol, int size)
{
  QPainterPath path;
  qreal d = 0.5 * size;
  switch (symbol)
  {
    case NoSymbol:
      break;
      
    case Ellipse:
      path.addEllipse(-d,-d,2*d,2*d);
      break;
      
    case Rect:
      path.addRect(-d,-d,2*d,2*d);
      break;
      
    case Diamond:
      path.addRect(-d,-d,2*d,2*d);
      path = QTransform().rotate(45).map(path);
      break;
      
    case Triangle:
    case UTriangle:
      path = trianglePath(d, 0);
      break;
      
    case DTriangle:
      path = trianglePath(d, 180);
      break;
      
    case LTriangle:
      path = trianglePath(d, -90);
      break;
    
    case RTriangle:
      path = trianglePath(d, 90);
      break;

    case Cross:
      path = crossPath(d, 0);
      break;
    
    case XCross:
      path = crossPath(d, 45);
      break;
      
    case HLine:
      path.moveTo(-d,0);
      path.lineTo(d,0);
      break;
      
    case VLine:
      path.moveTo(0,-d);
      path.lineTo(0,d);
      break;
      
    case Star1:
      path.addPath(crossPath(d,0));
      path.addPath(crossPath(d,45));
      break;
      
    case Star2:
      path = hexPath(d, true);
      break;
      
    case Hexagon:
      path = hexPath(d, false);
      break;
      
    default:
     // qWarning() << "Unsupported symbol" << symbol;
        break;
  }
  return path;
}


QPainterPath Point::trianglePath(double d, double rot) {
    QPainterPath path;
    path.moveTo(-d, d*sqrt(3.0)/3);
    path.lineTo(d, d*sqrt(3.0)/3);
    path.lineTo(0, -2*d*sqrt(3.0)/3);
    path.closeSubpath();
    return QTransform().rotate(rot).map(path);
}

QPainterPath Point::crossPath(double d, double rot)
{
    QPainterPath path;
    path.lineTo(0,d);
    path.moveTo(0,0);
    path.lineTo(0,-d);
    path.moveTo(0,0); 
    path.lineTo(d,0);
    path.moveTo(0,0);
    path.lineTo(-d,0);
    return QTransform().rotate(rot).map(path);
}

QPainterPath Point::hexPath(double d, bool star) {
    QPainterPath path;
    if (!star)
    {
        path.moveTo(d,0);
    }
    for (int i = 0; i < 6; ++i)
    {
        path.lineTo( d * cos(M_PI/3*i), d*sin(M_PI/3*i) );
        if (star)
        {
            path.lineTo(0,0);
        }
    }
    path.closeSubpath();
    return path;
}

QPixmap Point::pixmap_for_symbol(int symbol, QColor color, int size)
{
    // Indexed8 is the only format with a color table, which means we can replace entire colors
    // and not only indididual pixels
    QPixmap image(QSize(size, size));
    
    // TODO: Create fils with actual images, preferably SVG so they are scalable
    // image.load(filename);
    return image;
}

QRectF Point::rect_for_size(double size)
{
    return QRectF(-size/2, -size/2, size, size);
}

void Point::set_state(Point::State state) 
{
    m_state = state;
}

Point::State Point::state() const 
{
    return m_state;
}

void Point::set_state_flag(Point::StateFlag flag, bool on) 
{
    if (on)
    {
        m_state |= flag;
    }
    else
    {
        m_state &= ~flag;
    }

    if ((flag == Selected || flag == Marked) && label && ((Curve *)parentObject())->labels_on_marked())
    {
		if (is_selected() || is_marked())
		{
			label->show();
		}
		else
		{
			label->hide();
		}
    }

    update();
}

bool Point::state_flag(Point::StateFlag flag) const 
{
    return m_state & flag;
}

void Point::set_selected(bool selected)
{
    set_state_flag(Selected, selected);
}

bool Point::is_selected() const
{
    return state_flag(Selected);
}

void Point::set_marked(bool marked)
{
    set_state_flag(Marked, marked);
}

bool Point::is_marked() const
{
    return state_flag(Marked);
}

bool Point::is_transparent()
{
    return m_transparent;
}
void Point::set_transparent(bool transparent)
{
    m_transparent = transparent;
}

DataPoint Point::coordinates() const
{
    return m_coordinates;
}

void Point::set_coordinates(const DataPoint& data_point)
{
    m_coordinates = data_point;
}

void Point::clear_cache()
{
    pixmap_cache.clear();
}

/*
void Point::set_label(const QString& label)
{
    m_label = label;
}
*/

QString Point::text() const
{
    if (label == NULL)
    {
        return "";
    }
    else
    {
        return label->toPlainText();
    }
}

#include "point.moc"
