#ifndef CANVAS_3D_H
#define CANVAS_3D_H

#include "triple.h"

#include <deque>
#include <algorithm>

#include <QtGui/QWidget>
#include <QtCore/QDebug>
#include <QtCore/QMap>
#include <QtCore/QList>

#ifdef _WIN32
#define NOMINMAX // Avoiding clashing with std::numeric_limits
#include <windows.h> // Errors in gl.h when not included (VS10)
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

class Edge3D;

class Node3D
{
public:
    Node3D(int index, int symbol, QColor color, int size);
    virtual ~Node3D();

    void set_coordinates(double x, double y, double z=0.5);
    Triple<double, double, double> coordinates() const;

    void set_x(double x);
    double x() const;

    void set_y(double y);
    double y() const;

    void set_z(double z);
    double z() const;

    void set_index(int index);
    int index() const;

    void set_size(double size);
    double size() const;

    void set_tooltip(const QString& tooltip);
    QString tooltip() const;

    void set_marked(bool marked);
    bool marked() const;

    void set_selected(bool selected);
    bool selected() const;

    void set_label(const QString& label);
    QString label() const;

    void set_uuid(int uuid);
    int uuid() const;

    void set_color(const QColor& color);
    QColor color() const;

    QList<Node3D*> neighbors();

    /**
     * @brief Connect an edge to this node
     * 
     * A connected edge is automatically updated whenever this node is moved
     *
     * @param edge the edge to be connected
     **/
    void add_connected_edge(Edge3D* edge);
    void remove_connected_edge(Edge3D* edge);
    QList<Edge3D*> connected_edges();

private:
    double m_x;
    double m_y;
    double m_z;

    int m_index;
    double m_size;
    int m_uuid;

    bool m_marked;
    bool m_selected;
    QString m_label;

    QList<Edge3D*> m_connected_edges;
    QString m_tooltip;
    QColor m_color;
};

class Edge3D
{
public:
    enum Arrow
    {
        ArrowU = 0x01,
        ArrowV = 0x02
    };
    Q_DECLARE_FLAGS(Arrows, Arrow)

    Edge3D(Node3D* u, Node3D* v);
    virtual ~Edge3D();

    void set_u(Node3D* item);
    Node3D* u();
    void set_v(Node3D* item);
    Node3D* v();

    void set_label(const QString& label);
    QString label() const;
    void set_tooltip(const QString& tooltip);

    void set_links_index(int index);
    int links_index() const;

    void set_weight(double weight);
    double weight() const;

    void set_arrows(Arrows arrows);
    void set_arrow(Arrow arrow, bool enable);
    Arrows arrows();

private:
    Arrows m_arrows;
    Node3D* m_u;
    Node3D* m_v;
    int m_links_index;
    double m_weight;
    QString m_label;
};

class Canvas3D : public QWidget
{
    Q_OBJECT
public:
	enum CircularLayoutType
	{
		circular_original = 0x01,
		circular_random = 0x02,
		circular_crossing = 0x03
	};

    explicit Canvas3D(QWidget* parent = 0);
    virtual ~Canvas3D();

    typedef QList<Edge3D*> Edges;
    typedef QMap<int, Node3D*> Nodes;

    int random();
    int circular(CircularLayoutType type);
    int circular_crossing_reduction();
    int fr(int steps, bool weighted, bool smooth_cooling);

    Nodes nodes() const;
    void set_nodes(const Nodes& nodes);
    void add_nodes(const Nodes& nodes);
    void remove_node(int index);
    void remove_nodes(const QList< int >& nodes);

    Edges edges() const;
    void set_edges(const Edges& edges);
    void add_edges(const Edges& edges);

    QList<QPair<int, int> > edge_indices();

    void set_node_colors(const QMap<int, QColor>& colors);
    void set_node_sizes(const QMap<int, double>& sizes, double min_size, double max_size);
    void set_node_labels(const QMap<int, QString>& labels);
    void set_node_tooltips(const QMap<int, QString>& tooltips);
    void set_node_marks(const QMap<int, bool>& marks);
    void clear_node_marks();
    void set_node_coordinates(const QMap<int, Triple<double, double, double> >& coordinates);

    void set_edge_colors(const QList<QColor>& colors);
    void set_edge_sizes(double max_size);
    void set_edge_labels(const QList<QString>& labels);

    void set_min_node_size(double size);
    double min_node_size() const;

    void set_max_node_size(double size);
    double max_node_size() const;

    void set_use_animations(bool use_animations);
    bool use_animations() const;

    void set_labels_on_marked_only(bool labels_on_marked_only);
    bool labels_on_marked_only();

    void set_show_component_distances(bool show_component_distances);

    void stop_optimization();

    QRectF data_rect() const;

    void update();
    void draw_edges();
    void draw_nodes();

private:
    Nodes m_nodes;
    Edges m_edges;

    double m_min_node_size;
    double m_max_node_size;
    bool m_use_animations;
    bool m_stop_optimization;
    bool m_labels_on_marked_only;
    bool m_show_component_distances;

    GLuint m_vbo_edges;
    GLuint m_vbo_nodes;
    bool m_vbos_generated;
};

#endif
