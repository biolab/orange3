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

#ifndef NETWORKCURVE_H
#define NETWORKCURVE_H

#include "curve.h"
#include "point.h"
#include "plot.h"
#include <deque>
#include <algorithm>

class QueueVertex
{
public:
	int ndx;
	int position;
	unsigned int unplacedNeighbours;
	unsigned int placedNeighbours;
	std::vector<int> neighbours;

	bool hasNeighbour(int index)
	{
		std::vector<int>::iterator iter;

		for (iter = neighbours.begin(); iter != neighbours.end(); iter++)
			if (*iter == index)
				return true;

		return false;
	}

	friend std::ostream & operator<<(std::ostream &os, const QueueVertex &v)
	{
		os << "ndx: " << v.ndx << " unplaced: " << v.unplacedNeighbours << " placed: " << v.placedNeighbours << " neighbours: ";
		int i;
		for (i = 0; i < v.neighbours.size(); i++)
			os << v.neighbours[i] << " ";

		return (os);
	}

	QueueVertex(int index = -1, unsigned int neighbours = 0)
	{
		ndx = index;
		unplacedNeighbours = neighbours;
		placedNeighbours = 0;
	}

	bool operator () (const QueueVertex * a, const QueueVertex * b)
	{
		if (a->unplacedNeighbours < b->unplacedNeighbours)
			return false;
		else if (a->unplacedNeighbours > b->unplacedNeighbours)
			return true;
		else
		{
			return a->placedNeighbours < b->placedNeighbours;
		}
	}
};

class EdgeItem;

class NodeItem : public Point
{
public:
    enum {Type = Point::Type + 1};
    NodeItem(int index, int symbol, QColor color, int size, QGraphicsItem* parent = 0);
    virtual ~NodeItem();

    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    virtual int type() const {return Type;}
    
    void set_coordinates(double x, double y);


    void set_x(double x);
    double x() const;

    void set_y(double y);
    double y() const;
    
    void set_image(QPixmap* im);

    virtual void set_graph_transform(const QTransform& transform);
    virtual QTransform graph_transform() const;
    
    void set_index(int index);
    int index() const;

    void set_tooltip(const QString& tooltip);

    void set_uuid(int uuid);
    int uuid() const;
    
    QList<NodeItem*> neighbors();

    /**
     * @brief Connect an edge to this node
     * 
     * A connected edge is automatically updated whenever this node is moved
     *
     * @param edge the edge to be connected
     **/
    void add_connected_edge(EdgeItem* edge);
    void remove_connected_edge(EdgeItem* edge);
    QList<EdgeItem*> connected_edges();
    
    double m_size_value;

protected:
    QPixmap *image;

private:
    double m_x;
    double m_y;
    
    int m_index;
    int m_uuid;
    
    QList<EdgeItem*> m_connected_edges;
    QTransform m_graph_transform;
};

class ModelItem : public NodeItem
{
public:
	ModelItem(int index, int symbol, QColor color, int size, QGraphicsItem* parent = 0);

    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);

    void set_representative(bool value);
    bool is_representative() const;

private:
    bool representative;
};

struct ArrowData
{
	ArrowData(int size, const QColor& color) : size(size), color(color) {}

	int size;
    QColor color;
};

class EdgeItem : public QAbstractGraphicsShapeItem
{
public:
    enum Arrow
    {
        ArrowU = 0x01,
        ArrowV = 0x02
    };
    Q_DECLARE_FLAGS(Arrows, Arrow)
    
    EdgeItem(NodeItem* u, NodeItem* v, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
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
    
    void set_arrows(Arrows arrows);
    void set_arrow(Arrow arrow, bool enable);
    Arrows arrows();
    
    static QHash<ArrowData, QPixmap> arrow_cache;

private:
    Arrows m_arrows;
    NodeItem* m_u;
    NodeItem* m_v;
    int m_links_index;
    double m_weight;
    QString m_label;
};

class NetworkCurve : public Curve
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
    typedef QMap<int, LabelItem*> Labels;

    explicit NetworkCurve(QGraphicsItem* parent = 0);
    virtual ~NetworkCurve();

    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget);

    virtual void update_properties();
    virtual QRectF data_rect() const;
    virtual void register_points();
    
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

private:
    void scale_axes();

    Nodes m_nodes;
    Edges m_edges;
    Labels m_labels;

    double m_min_node_size;
    double m_max_node_size;
    bool m_use_animations;
    bool m_stop_optimization;
    bool m_show_component_distances;
};

#endif // NETWORKCURVE_H
