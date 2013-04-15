// DataPoint is implemented as a Python 2-element tuple.
%MappedType DataPoint /DocType="tuple-of-float-float"/
{
%TypeHeaderCode
#include "point.h"
%End

%ConvertFromTypeCode
    // Create the tuple.
    return Py_BuildValue((char *)"dd", sipCpp->x, sipCpp->y);
%End

%ConvertToTypeCode
    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
        return (PyTuple_Size(sipPy) == 2);
    
    DataPoint* qp = new DataPoint();

    PyArg_ParseTuple(sipPy, "dd", &qp->x, &qp->y);
    *sipCppPtr = qp;
    
    return sipGetState(sipTransferObj);
%End
};

class Point : QGraphicsObject
{
    
%TypeHeaderCode
#include "point.h"
%End

public:
    enum DisplayMode
    {
        DisplayPixmap,
        DisplayPath
    };
    
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

    typedef QFlags<Point::StateFlag> State;
  
    Point(QGraphicsItem* parent /TransferThis/ = 0);
    Point(int symbol, QColor color, int size, QGraphicsItem* parent /TransferThis/ = 0);
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
    static void clear_cache();
};
