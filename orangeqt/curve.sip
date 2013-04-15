struct Updater
{

%TypeHeaderCode
#include "curve.h"
%End

    Updater(qreal scale, const QPen& pen, const QBrush& brush, const QPainterPath& path);
    
    void operator()(QGraphicsPathItem* item);
    
    qreal m_scale;
    QPen m_pen;
    QBrush m_brush;
};


typedef QList<DataPoint> Data;

class Curve : PlotItem {

%TypeHeaderCode
#include "curve.h"
%End

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

  Curve(const QList< double >& x_data, const QList< double >& y_data, QGraphicsItem* parent /TransferThis/ = 0);
  Curve(QGraphicsItem* parent /TransferThis/ = 0);
  virtual ~Curve();

  virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  virtual QRectF boundingRect() const;

  virtual void update_properties();
    
  Point* point_item(qreal x, qreal y, int size = 0, QGraphicsItem* parent = 0);
  
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
  void set_data(const QList<qreal>& x_data, const QList<qreal>& y_data);

  virtual QTransform graph_transform() const;
  virtual void set_graph_transform(const QTransform& transform);

  int style() const;
  void set_style(int style);
 
  virtual QTransform zoom_transform();
  virtual void set_zoom_transform(const QTransform& transform);

  bool auto_update() const;
  void set_auto_update(bool auto_update);

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
    UpdateAll = 0xFF
  };
  
  typedef QFlags<Curve::UpdateFlag> UpdateFlags;
  
  void set_dirty(Curve::UpdateFlags flags = Curve::UpdateAll);

  void set_points(const QList<Point*>& points);
  QList<Point*> points();
  
  void set_labels_on_marked(bool value);
  bool labels_on_marked();
  
protected:
  void set_updated(Curve::UpdateFlags flags);
  Curve::UpdateFlags needs_update();

  void cancel_all_updates();
  void update_number_of_items();

public slots:
    void update_point_coordinates();
    void update_point_positions();

};