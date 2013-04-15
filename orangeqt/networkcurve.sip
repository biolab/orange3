%ModuleCode
#include "networkcurve.h"
%End

class NodeItem : Point
{
public:
    NodeItem(int index, int symbol, QColor color, int size, QGraphicsItem* parent /TransferThis/ = 0);
    virtual ~NodeItem();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
    void set_coordinates(double x, double y);
    
    void set_x(double x);
    double x() const;
    
    void set_y(double y);
    double y() const;
    
    void set_graph_transform(const QTransform& transform);
    QTransform graph_transform() const;
    
    void set_index(int index);
    int index() const;

    void set_tooltip(const QString& tooltip);
    
    void set_uuid(int uuid);
    int uuid() const;
    
    void add_connected_edge(EdgeItem* edge);
    void remove_connected_edge(EdgeItem* edge);
    
    void set_image(QPixmap* im);
};
	
class ModelItem : NodeItem
{
public:
    ModelItem(int index, int symbol, QColor color, int size, QGraphicsItem* parent /TransferThis/ = 0);
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
    void set_representative(double y = true);
    bool is_representative() const;
};

class EdgeItem : QAbstractGraphicsShapeItem
{
public:
    enum Arrow
    {
        ArrowU = 0x01,
        ArrowV = 0x02
    };
    typedef QFlags<EdgeItem::Arrow> Arrows;
    
    EdgeItem(NodeItem* u, NodeItem* v, QGraphicsItem* parent /TransferThis/ = 0, QGraphicsScene* scene /TransferThis/ = 0);
    virtual ~EdgeItem();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    virtual QRectF boundingRect() const;
    virtual QPainterPath shape() const;
    
    void set_u(NodeItem* item);
    NodeItem* u();
    void set_v(NodeItem* item);
    NodeItem* v();
    
    void set_label(const QString& label);
    QString label() const;
    void set_tooltip(const QString& tooltip);
    
    void set_links_index(int index);
    int links_index() const;
    
    void set_weight(double weight);
    double weight() const;
    
    void set_arrows(EdgeItem::Arrows arrows);
    void set_arrow(EdgeItem::Arrow arrow, bool enable);
};

class NetworkCurve : Curve
{
public:
	enum CircularLayoutType
	{
		circular_original = 0x01,
		circular_random = 0x02,
		circular_crossing = 0x03
	};
	
    typedef QList<EdgeItem*> Edges;
    typedef QMap<int, NodeItem*> Nodes;
    typedef QMap<int, QGraphicsItem*> Labels;

    NetworkCurve(QGraphicsItem* parent /TransferThis/ = 0);
    virtual ~NetworkCurve();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
    virtual void update_properties();
    virtual QRectF data_rect() const;
    
    int random();
    int circular(NetworkCurve::CircularLayoutType type);
    int fr(int steps, bool weighted, bool smooth_cooling);

    Nodes nodes() const;
    void set_nodes(const Nodes& nodes);
    void add_nodes(const Nodes& nodes);
    void remove_node(int index);
    void remove_nodes(const QList< int >& nodes);
    
    Edges edges() const;
    void set_edges(const Edges& edges);
    void add_edges(const Edges& edges);
    
    Labels labels() const;
    void set_labels(const Labels& labels);
    void add_labels(const Labels& labels);
    void remove_label(int index);
    void remove_labels(const QList< int >& labels);
    
    QList<QPair<int, int> > edge_indices();

    void set_node_colors(const QMap<int, QColor>& colors);    
    void set_node_sizes(const QMap<int, double>& sizes, double min_size, double max_size);
    void set_node_labels(const QMap<int, QString>& labels);
    void set_node_tooltips(const QMap<int, QString>& tooltips);
    void set_node_marks(const QMap<int, bool>& marks);
    void clear_node_marks();
    void set_node_coordinates(const QMap<int, QPair<double, double> >& coordinates);
    
    void set_edge_colors(const QList<QColor>& colors);
    void set_edge_sizes(double max_size);
    void set_edge_labels(const QList<QString>& labels);
    
    void set_min_node_size(double size);
    double min_node_size() const;

    void set_max_node_size(double size);
    double max_node_size() const;
    
    void set_use_animations(bool use_animations);
    bool use_animations() const;
    
    void set_show_component_distances(bool show_component_distances);
    
    void stop_optimization();
};



