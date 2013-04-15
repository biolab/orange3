#include "canvas3d.h"

#include <limits>
#include <QtCore/QMap>
#include <QtCore/QList>
#include <QtCore/qmath.h>
#include <QCoreApplication>
#include <QtCore/QTime>

#include "glextensions.h"

#define PI 3.14159265

qreal _random()
{
    return ((qreal)(qrand()) / RAND_MAX);
}

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

Node3D::Node3D(int index, int symbol, QColor color, int size)
{
    set_index(index);
    set_coordinates(_random(), _random(), _random());
    set_size(1);
    set_marked(false);
    set_selected(false);
    set_label("");
}

Node3D::~Node3D()
{
}

void Node3D::set_coordinates(double x, double y, double z)
{
    m_x = x;
    m_y = y;
    m_z = z;
}

Triple<double, double, double> Node3D::coordinates() const
{
    return Triple<double, double, double>(m_x, m_y, m_z);
}

void Node3D::set_index(int index)
{
    m_index = index;
}

int Node3D::index() const
{
    return m_index;
}

void Node3D::set_size(double size)
{
    m_size = size;
}

double Node3D::size() const
{
    return m_size;
}

void Node3D::set_x(double x)
{
    set_coordinates(x, m_y);
}

double Node3D::x() const
{
    return m_x;
}

void Node3D::set_y(double y)
{
    set_coordinates(m_y, y);
}

double Node3D::y() const
{
    return m_y;
}

void Node3D::set_z(double z)
{
    set_coordinates(m_z, z);
}

double Node3D::z() const
{
    return m_z;
}

void Node3D::set_tooltip(const QString& tooltip)
{
    m_tooltip = tooltip;
}

QString Node3D::tooltip() const
{
    return m_tooltip;
}

void Node3D::set_uuid(int uuid)
{
    m_uuid = uuid;
}

int Node3D::uuid() const
{
    return m_uuid;
}

void Node3D::set_color(const QColor& color)
{
    m_color = color;
}

QColor Node3D::color() const
{
    return m_color;
}

void Node3D::set_marked(bool marked)
{
    m_marked = marked;
}

bool Node3D::marked() const
{
    return m_marked;
}

void Node3D::set_selected(bool selected)
{
    m_selected = selected;
}

bool Node3D::selected() const
{
    return m_selected;
}

void Node3D::set_label(const QString& label)
{
    m_label = label;
}

QString Node3D::label() const
{
    return m_label;
}

void Node3D::add_connected_edge(Edge3D* edge)
{
    if (!m_connected_edges.contains(edge))
    {
        m_connected_edges << edge;
    }
}

void Node3D::remove_connected_edge(Edge3D* edge)
{
    m_connected_edges.removeAll(edge);
}

QList<Edge3D*> Node3D::connected_edges()
{
	return m_connected_edges;
}

QList<Node3D*> Node3D::neighbors()
{
	QList<Node3D*> neighbors;
	Edge3D *e;
	QList<Edge3D*> edges = connected_edges();
	int size = edges.size();
	foreach(e, edges)
	{
		if (e->u()->index() == index())
		{
			neighbors.append(e->v());
		}
		else
		{
			neighbors.append(e->u());
		}
	}

	return neighbors;
}

Edge3D::Edge3D(Node3D* u, Node3D* v): m_u(0), m_v(0)
{
    set_u(u);
    set_v(v);
}

Edge3D::~Edge3D()
{
    if (m_u)
        m_u->remove_connected_edge(this);
    if (m_v)
        m_v->remove_connected_edge(this);
}

void Edge3D::set_u(Node3D* item)
{
    if (m_u)
        m_u->remove_connected_edge(this);
    if (item)
        item->add_connected_edge(this);
    m_u = item;
}

Node3D* Edge3D::u()
{
    return m_u;
}

void Edge3D::set_v(Node3D* item)
{
    if (m_v)
        m_v->remove_connected_edge(this);
    if (item)
        item->add_connected_edge(this);
    m_v = item;
}

Node3D* Edge3D::v()
{
    return m_v;
}

void Edge3D::set_tooltip(const QString& tooltip)
{
    //setToolTip(tooltip);
    //TODO
}

void Edge3D::set_label(const QString& label)
{
    m_label = label;
}

QString Edge3D::label() const
{
    return m_label;
}

void Edge3D::set_links_index(int index)
{
    m_links_index = index;
}

int Edge3D::links_index() const
{
    return m_links_index;
}

void Edge3D::set_arrow(Edge3D::Arrow arrow, bool enable)
{
    if (enable)
        set_arrows(arrows() | arrow);
    else
        set_arrows(arrows() & ~arrow);
}

Edge3D::Arrows Edge3D::arrows()
{
    return m_arrows;
}

void Edge3D::set_arrows(Edge3D::Arrows arrows)
{
    m_arrows = arrows;
}

void Edge3D::set_weight(double weight)
{
    m_weight = weight;
}

double Edge3D::weight() const
{
    return m_weight;
}

Canvas3D::Canvas3D(QWidget* parent) : QWidget(parent)
{
	 m_min_node_size = 5;
	 m_max_node_size = 5;
     m_vbos_generated = false;
}

Canvas3D::~Canvas3D()
{
    qDeleteAll(m_edges);
    m_edges.clear();
    qDeleteAll(m_nodes);
    m_nodes.clear();
}

int Canvas3D::random()
{
	Nodes::ConstIterator uit = m_nodes.constBegin();
	Nodes::ConstIterator uend = m_nodes.constEnd();

	for (; uit != uend; ++uit)
		uit.value()->set_coordinates(_random(), _random(), _random());

	return 0;
}

QRectF Canvas3D::data_rect() const
{
    QRectF r;
    bool first = true;
    foreach (const Node3D* node, m_nodes)
    {
        if (first)
        {
            r = QRectF(node->x(), node->y(), 0, 0);
            first = false;
        }
        else
        {
            r.setTop(qMin(r.top(), node->y()));
            r.setBottom(qMax(r.bottom(), node->y()));
            r.setLeft(qMin(r.left(), node->x()));
            r.setRight(qMax(r.right(), node->x()));
        }
    }

    return r;
}

int Canvas3D::circular(CircularLayoutType type)
{
	// type
	// 0 - original
	// 1 - random
	// 2 - crossing reduction

	if (type == Canvas3D::circular_crossing)
	{
		qDebug() << "crossing_reduction";
		return circular_crossing_reduction();
	}

	if (type == Canvas3D::circular_original)
		qDebug() << "original";

	if (type == Canvas3D::circular_random)
			qDebug() << "random";

	QRectF rect = data_rect();
	int xCenter = rect.width() / 2;
	int yCenter = rect.height() / 2;
	int r = (rect.width() < rect.height()) ? rect.width() * 0.38 : rect.height() * 0.38;

	int i;
	double fi = PI;
	double step = 2 * PI / m_nodes.size();

	qsrand(QTime(0,0,0).secsTo(QTime::currentTime()));
	std::vector<int> vertices;
	Nodes::ConstIterator it;
	for (it = m_nodes.constBegin(); it != m_nodes.constEnd(); ++it)
	{
		vertices.push_back(it.key());
	}

	for (i = 0; i < m_nodes.size(); ++i)
	{
		if (type == Canvas3D::circular_original)
		{
			m_nodes[vertices[i]]->set_coordinates(r * cos(fi) + xCenter, r * sin(fi) + yCenter);
		}
		else if (type == Canvas3D::circular_random)
		{
			int ndx = rand() % vertices.size();
                        m_nodes[vertices[ndx]]->set_coordinates(r * cos(fi) + xCenter, r * sin(fi) + yCenter);
			vertices.erase(vertices.begin() + ndx);
		}
		fi = fi - step;
	}
	//register_points();
	return 0;
}

int Canvas3D::circular_crossing_reduction()
{
	QMap<int, QueueVertex*> qvertices;
	std::vector<QueueVertex*> vertices;
	std::vector<QueueVertex*> original;

	Nodes::ConstIterator it;
	for (it = m_nodes.constBegin(); it != m_nodes.constEnd(); ++it)
	{
		QueueVertex *vertex = new QueueVertex();
		vertex->ndx = it.key();
		qvertices[it.key()] = vertex;

		std::vector<int> neighbours;
		vertex->unplacedNeighbours = neighbours.size();
		vertex->neighbours = neighbours;
		vertices.push_back(vertex);
	}
	int i;
	Edge3D *edge;
	for (i = 0; i < m_edges.size(); i++)
	{
		edge = m_edges[i];
		int u = edge->u()->index();
		int v = edge->v()->index();
		qvertices[u]->neighbours.push_back(v);
		qvertices[u]->unplacedNeighbours += 1;
		qvertices[v]->neighbours.push_back(u);
		qvertices[v]->unplacedNeighbours += 1;
	}
	original.assign(vertices.begin(), vertices.end());

	std::deque<int> positions;
	while (vertices.size() > 0)
	{
		std::sort(vertices.begin(), vertices.end(), QueueVertex());
		QueueVertex *vertex = vertices.back();

		// update neighbours
		for (i = 0; i < vertex->neighbours.size(); ++i)
		{
			int ndx = vertex->neighbours[i];

			original[ndx]->placedNeighbours++;
			original[ndx]->unplacedNeighbours--;
		}

		// count left & right crossings
		if (vertex->placedNeighbours > 0)
		{
			int left = 0;
			std::vector<int> lCrossings;
			std::vector<int> rCrossings;
			for (i = 0; i < positions.size(); ++i)
			{
				int ndx = positions[i];

				if (vertex->hasNeighbour(ndx))
				{
					lCrossings.push_back(left);
					left += original[ndx]->unplacedNeighbours;
					rCrossings.push_back(left);
				}
				else
					left += original[ndx]->unplacedNeighbours;
			}

			int leftCrossings = 0;
			int rightCrossings = 0;

			for (i = 0; i < lCrossings.size(); --i)
				leftCrossings += lCrossings[i];

			rCrossings.push_back(left);
			for (i = rCrossings.size() - 1; i > 0 ; --i)
				rightCrossings += rCrossings[i] - rCrossings[i - 1];
			//cout << "left: " << leftCrossings << " right: " <<rightCrossings << endl;
			if (leftCrossings < rightCrossings)
				positions.push_front(vertex->ndx);
			else
				positions.push_back(vertex->ndx);
		}
		else
			positions.push_back(vertex->ndx);

		vertices.pop_back();
	}

	// Circular sifting
	for (i = 0; i < positions.size(); ++i)
		original[positions[i]]->position = i;

	int step;
	for (step = 0; step < 5; ++step)
	{
		for (i = 0; i < m_nodes.size(); ++i)
		{
			bool stop = false;
			int switchNdx = -1;
			QueueVertex *u = original[positions[i]];
			int vNdx = (i + 1) % m_nodes.size();

			while (!stop)
			{
				QueueVertex *v = original[positions[vNdx]];

				int midCrossings = u->neighbours.size() * v->neighbours.size() / 2;
				int crossings = 0;
				int j,k;
				for (j = 0; j < u->neighbours.size(); ++j)
					for (k = 0; k < v->neighbours.size(); ++k)
						if ((original[u->neighbours[j]]->position == v->position) || (original[v->neighbours[k]]->position == u->position))
							midCrossings = (u->neighbours.size() - 1) * (v->neighbours.size() - 1) / 2;
						else if ((original[u->neighbours[j]]->position + m_nodes.size() - u->position) % m_nodes.size() < (original[v->neighbours[k]]->position + m_nodes.size() - u->position) % m_nodes.size())
							++crossings;

				//cout << "v: " <<  v->ndx << " crossings: " << crossings << " u.n.size: " << u->neighbours.size() << " v.n.size: " << v->neighbours.size() << " mid: " << midCrossings << endl;
				if (crossings > midCrossings)
					switchNdx = vNdx;
				else
					stop = true;

				vNdx = (vNdx + 1) % m_nodes.size();
			}
			int j;
			if (switchNdx > -1)
			{
				//cout << "u: " << u->ndx << " switch: " << original[switchNdx]->ndx << endl << endl;
				positions.erase(positions.begin() + i);
				positions.insert(positions.begin() + switchNdx, u->ndx);

				for (j = i; j <= switchNdx; ++j)
					original[positions[j]]->position = j;
			}
			//else
			//	cout << "u: " << u->ndx << " switch: " << switchNdx << endl;
		}
	}

	QRectF rect = data_rect();
	int xCenter = rect.width() / 2;
	int yCenter = rect.height() / 2;
	int r = (rect.width() < rect.height()) ? rect.width() * 0.38 : rect.height() * 0.38;
	double fi = PI;
	double fiStep = 2 * PI / m_nodes.size();

	for (i = 0; i < m_nodes.size(); ++i)
	{
		m_nodes[positions[i]]->set_x(r * cos(fi) + xCenter);
		m_nodes[positions[i]]->set_y(r * sin(fi) + yCenter);
		fi = fi - fiStep;
	}

    qDeleteAll(original);

	original.clear();
	vertices.clear();
	qvertices.clear();
        
    //register_points();
        
	return 0;
}

int Canvas3D::fr(int steps, bool weighted, bool smooth_cooling)
{
	int i, j;
	int count = 0;
	Node3D *u, *v;
	Edge3D *edge;
	m_stop_optimization = false;

	double rect[4] = {std::numeric_limits<double>::max(),
			  std::numeric_limits<double>::max(),
			  std::numeric_limits<double>::min(),
			  std::numeric_limits<double>::min()};

	QMap<int, Triple<double, double, double> > disp;
	foreach (const Node3D* node, m_nodes)
	{
		disp[node->index()] = node->coordinates();

		double x = node->x();
		double y = node->y();

		if (rect[0] > x) rect[0] = x;
		if (rect[1] > y) rect[1] = y;
		if (rect[2] < x) rect[2] = x;
		if (rect[3] < y) rect[3] = y;
	}

	QRectF data_r(rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]);
	double area = data_r.width() * data_r.height();
	double k2 = area / m_nodes.size();
	double k = sqrt(k2);
	double kk = 2 * k;
	double kk2 = kk * kk;
	double jitter = sqrt(area) / 2000;
	double temperature, cooling, cooling_switch, cooling_1, cooling_2;
	temperature = sqrt(area) / 5;
	cooling = exp(log(k / 10 / temperature) / steps);

	if (steps > 20)
	{
		cooling_switch = sqrt(area) / 100;
		cooling_1 = (temperature - cooling_switch) / 20;
		cooling_2 = (cooling_switch - sqrt(area) / 2000 ) / (steps - 20);
	}
	else
	{
		cooling_switch = sqrt(area) / 1000;
		cooling_1 = (temperature - cooling_switch) / steps;
		cooling_2 = 0;
	}

	if (smooth_cooling)
	{
		if (steps < 20)
		{
			steps = 20;
		}
		temperature = cooling_switch;
		cooling_1 = 0;
		cooling_2 = (cooling_switch - sqrt(area) / 2000 ) / steps;
	}

	// iterations
	//clock_t refresh_time = clock() + 0.05 * CLOCKS_PER_SEC;
	//Plot *p = plot();
	//bool animation_enabled = p->animate_points;
	//p->animate_points = false;

	QTime refresh_time = QTime::currentTime();
	for (i = 0; i < steps; ++i)
	{
		foreach (const Node3D* node, m_nodes)
		{
			disp[node->index()].first = 0;
			disp[node->index()].second = 0;
		}

		// calculate repulsive force
		Nodes::ConstIterator uit = m_nodes.constBegin();
		Nodes::ConstIterator uend = m_nodes.constEnd();
		for (; uit != uend; ++uit)
		{
			u = uit.value();
			Nodes::ConstIterator vit(uit);
			++vit;
			for (; vit != uend; ++vit)
			{
				v = vit.value();

				double difx = u->x() - v->x();
				double dify = u->y() - v->y();

				double dif2 = difx * difx + dify * dify;

				// if nodes are close, apply repulsive force
				if (dif2 < kk2)
				{
					// if nodes overlap
					if (dif2 == 0)
					{
						dif2 = 1 / k;
						u->set_x(u->x() + jitter);
						u->set_y(u->y() + jitter);
						v->set_x(v->x() - jitter);
						v->set_y(v->y() - jitter);
					}

					double dX = difx * k2 / dif2;
					double dY = dify * k2 / dif2;

					disp[u->index()].first += dX;
					disp[u->index()].second += dY;

					disp[v->index()].first -= dX;
					disp[v->index()].second -= dY;
				}
			}
		}
		// calculate attractive forces
		for (j = 0; j < m_edges.size(); ++j)
		{
			edge = m_edges[j];
			double difx = edge->u()->x() - edge->v()->x();
			double dify = edge->u()->y() - edge->v()->y();

			double dif = sqrt(difx * difx + dify * dify);

			double dX = difx * dif / k;
			double dY = dify * dif / k;

			if (weighted) {
				dX *= edge->weight();
				dY *= edge->weight();
			}

			disp[edge->u()->index()].first -= dX;
			disp[edge->u()->index()].second -= dY;

			disp[edge->v()->index()].first += dX;
			disp[edge->v()->index()].second += dY;
		}

		// limit the maximum displacement to the temperature t
		// and then prevent from being displaced outside frame
		Nodes::Iterator nit = m_nodes.begin();
		for (; nit != m_nodes.end(); ++nit)
		{
			u = nit.value();
			double dif = sqrt(disp[u->index()].first * disp[u->index()].first + disp[u->index()].second * disp[u->index()].second);

			if (dif == 0)
				dif = 1;

			u->set_coordinates(u->x() + (disp[u->index()].first * qMin(fabs(disp[u->index()].first), temperature) / dif),
			                   u->y() + (disp[u->index()].second * qMin(fabs(disp[u->index()].second), temperature) / dif));
		}

		QTime before_refresh_time = QTime::currentTime();
		if (before_refresh_time > refresh_time && i % 2 == 0)
		{
			//update_properties();
            QCoreApplication::processEvents();
			int refresh_duration = before_refresh_time.msecsTo(QTime::currentTime());

			refresh_time = before_refresh_time.addMSecs(qMax(refresh_duration * 3, 10));
		}
		if (m_stop_optimization)
		{
			break;
		}
		if (floor(temperature) > cooling_switch)
		{
			temperature -= cooling_1;
		}
		else
		{
			temperature -= cooling_2;
		}
	}

	//p->animate_points = animation_enabled;
	//register_points();
	return 0;
}

void Canvas3D::set_edges(const Canvas3D::Edges& edges)
{
    //cancel_all_updates();
    qDeleteAll(m_edges);
    m_edges = edges;
}

Canvas3D::Edges Canvas3D::edges() const
{
    return m_edges;
}

QList<QPair<int, int> > Canvas3D::edge_indices()
{
	int i;
	Edge3D *e;
	QList<QPair<int, int> > edge_indices;

	for (i = 0; i < m_edges.size(); ++i)
	{
		e = m_edges[i];
		edge_indices.append(QPair<int, int>(e->u()->index(), e->v()->index()));
	}

	return edge_indices;
}

void Canvas3D::set_nodes(const Canvas3D::Nodes& nodes)
{
    //cancel_all_updates();
    qDeleteAll(m_edges);
    m_edges.clear();
    qDeleteAll(m_nodes);
    m_nodes = nodes;
    Q_ASSERT(m_nodes.uniqueKeys() == m_nodes.keys());
    //register_points();
}

Canvas3D::Nodes Canvas3D::nodes() const
{
    return m_nodes;
}

void Canvas3D::remove_nodes(const QList<int>& nodes)
{
    //cancel_all_updates();
    foreach (int i, nodes)
    {
        remove_node(i);
    }
}

void Canvas3D::remove_node(int index)
{
    //cancel_all_updates();
    if (!m_nodes.contains(index))
    {
        qWarning() << "Trying to remove node" << index << "which is not in the network";
        return;
    }
    Node3D* node = m_nodes.take(index);
    Q_ASSERT(node->index() == index);
    /*Plot* p = plot();
    if (p)
    {
        p->remove_point(node, this);
    }*/

    foreach (Edge3D* edge, node->connected_edges())
    {
        m_edges.removeOne(edge);
        delete edge;
    }
    Q_ASSERT(node->connected_edges().isEmpty());
    delete node;
}

void Canvas3D::add_edges(const Canvas3D::Edges& edges)
{
    m_edges.append(edges);
}

void Canvas3D::add_nodes(const Canvas3D::Nodes& nodes)
{
    Nodes::ConstIterator it = nodes.constBegin();
    Nodes::ConstIterator end = nodes.constEnd();
    QList<int> indices;
	for (it; it != end; ++it)
	{
		indices.append(it.key());

		if (m_nodes.contains(it.key()))
		{
			remove_node(it.key());
		}
	}

	m_nodes.unite(nodes);
    Q_ASSERT(m_nodes.uniqueKeys() == m_nodes.keys());
	//register_points();
}

void Canvas3D::set_node_colors(const QMap<int, QColor>& colors)
{
	QMap<int, QColor>::ConstIterator it;
	for (it = colors.constBegin(); it != colors.constEnd(); ++it)
	{
		m_nodes[it.key()]->set_color(it.value());
	}
}

void Canvas3D::set_node_sizes(const QMap<int, double>& sizes, double min_size, double max_size)
{
    //cancel_all_updates();

	Node3D* node;
	Nodes::ConstIterator nit;

	double min_size_value = std::numeric_limits<double>::max();
	double max_size_value = std::numeric_limits<double>::min();

	QMap<int, double>::ConstIterator it;
	for (it = sizes.begin(); it != sizes.end(); ++it)
	{
		m_nodes[it.key()]->set_size(it.value());

		if (it.value() < min_size_value)
		{
			min_size_value = it.value();
		}

		if (it.value() > max_size_value)
		{
			max_size_value = it.value();
		}
	}

	// find min and max size value in nodes dict
	bool min_changed = true;
	bool max_changed = true;
	for (nit = m_nodes.constBegin(); nit != m_nodes.constEnd(); ++nit)
	{
		node = nit.value();

		if (node->size() < min_size_value)
		{
			min_size_value = node->size();
			min_changed = false;
		}

		if (node->size() > max_size_value)
		{
			max_size_value = node->size();
			max_changed = false;
		}
	}

	double size_span = max_size_value - min_size_value;

	if (min_size > 0 || max_size > 0 || min_changed || max_changed)
	{
		if (min_size > 0)
		{
			m_min_node_size = min_size;
		}

		if (max_size > 0)
		{
			m_max_node_size = max_size;
		}

		double node_size_span = m_max_node_size - m_min_node_size;
		// recalibrate all
		if (size_span > 0)
		{
			for (nit = m_nodes.constBegin(); nit != m_nodes.constEnd(); ++nit)
			{
				node = nit.value();
				node->set_size((node->size() - min_size_value) / size_span * node_size_span + m_min_node_size);
			}
		}
		else
		{
			for (nit = m_nodes.constBegin(); nit != m_nodes.constEnd(); ++nit)
			{
				node = nit.value();
				node->set_size(m_min_node_size);
			}
		}
	}
	else if (sizes.size() > 0)
	{
		double node_size_span = m_max_node_size - m_min_node_size;
		// recalibrate given
		if (size_span > 0)
		{
			for (it = sizes.begin(); it != sizes.end(); ++it)
			{
				node = m_nodes[it.key()];
				node->set_size((node->size() - min_size_value) / size_span * node_size_span + m_min_node_size);
			}
		}
		else
		{
			for (it = sizes.begin(); it != sizes.end(); ++it)
			{
				node = m_nodes[it.key()];
				node->set_size(m_min_node_size);
			}
		}
	}
}

void Canvas3D::set_node_labels(const QMap<int, QString>& labels)
{
    //cancel_all_updates();
	QMap<int, QString>::ConstIterator it;
	for (it = labels.constBegin(); it != labels.constEnd(); ++it)
	{
		m_nodes[it.key()]->set_label(it.value());
	}
}

void Canvas3D::set_node_tooltips(const QMap<int, QString>& tooltips)
{
    //cancel_all_updates();
	QMap<int, QString>::ConstIterator it;
	for (it = tooltips.constBegin(); it != tooltips.constEnd(); ++it)
	{
		m_nodes[it.key()]->set_tooltip(it.value());
	}
}

void Canvas3D::set_node_marks(const QMap<int, bool>& marks)
{
	QMap<int, bool>::ConstIterator it;
	for (it = marks.constBegin(); it != marks.constEnd(); ++it)
	{
		m_nodes[it.key()]->set_marked(it.value());
	}
}

void Canvas3D::clear_node_marks()
{
	Nodes::Iterator it;
	for (it = m_nodes.begin(); it != m_nodes.end(); ++it)
	{
		it.value()->set_marked(false);
	}
}

void Canvas3D::set_node_coordinates(const QMap<int, Triple<double, double, double> >& coordinates)
{
	Node3D *node;
	QMap<int, Triple<double, double, double> >::ConstIterator it = coordinates.constBegin();
	for (; it != coordinates.constEnd(); ++it)
	{
		node = m_nodes[it.key()];
		node->set_x(it.value().first);
		node->set_y(it.value().second);
		node->set_z(it.value().third);
	}
}

void Canvas3D::set_edge_colors(const QList<QColor>& colors)
{
	int i;
	/*for (i = 0; i < colors.size(); ++i)
	{
		QPen p = m_edges[i]->pen();
		p.setColor(colors[i]);
		m_edges[i]->setPen(p);
	}*/
}

void Canvas3D::set_edge_sizes(double max_size)
{
    double min_size_value = std::numeric_limits<double>::max();
	double max_size_value = std::numeric_limits<double>::min();

	int i;
	for (i = 0; i < m_edges.size(); ++i)
	{
		double w = m_edges[i]->weight();
		if (w < min_size_value)
		{
			min_size_value = w;
		}
		if (w > max_size_value)
		{
			max_size_value = w;
		}
	}

	double size_span = max_size_value - min_size_value;
	double edge_size_span = (max_size > 0) ? max_size - 1 : 0;

	if (size_span > 0 && edge_size_span > 0)
	{
		for (i = 0; i < m_edges.size(); ++i)
		{
			double w = m_edges[i]->weight();
			/*QPen p = m_edges[i]->pen();
			p.setWidthF((w - min_size_value) / size_span * edge_size_span + 1);
			m_edges[i]->setPen(p);*/
		}
	}
	else
	{
		for (i = 0; i < m_edges.size(); ++i)
		{
			/*QPen p = m_edges[i]->pen();
			p.setWidthF(1);
			m_edges[i]->setPen(p);*/
		}
	}
}

void Canvas3D::set_edge_labels(const QList<QString>& labels)
{
	int i;
	for (i = 0; i < labels.size(); ++i)
	{
		m_edges[i]->set_label(labels[i]);
	}
}

void Canvas3D::set_min_node_size(double size)
{
	set_node_sizes(QMap<int, double>(), size, 0);
}

double Canvas3D::min_node_size() const
{
	return m_min_node_size;
}

void Canvas3D::set_max_node_size(double size)
{
	set_node_sizes(QMap<int, double>(), 0, size);
}

double Canvas3D::max_node_size() const
{
	return m_max_node_size;
}

void Canvas3D::set_use_animations(bool use_animations)
{
    m_use_animations = use_animations;
}

bool Canvas3D::use_animations() const
{
    return m_use_animations;
}

void Canvas3D::set_labels_on_marked_only(bool labels_on_marked_only)
{
	m_labels_on_marked_only = labels_on_marked_only;
}

bool Canvas3D::labels_on_marked_only()
{
	return m_labels_on_marked_only;
}

void Canvas3D::set_show_component_distances(bool show_component_distances)
{
	m_show_component_distances = show_component_distances;
}
 
void Canvas3D::stop_optimization()
{
    m_stop_optimization = true;
}

void Canvas3D::update()
{
    if (m_nodes.size() == 0)
        return;

    init_gl_extensions();

    if (m_vbos_generated)
    {
        glDeleteBuffers(1, &m_vbo_edges);
        glDeleteBuffers(1, &m_vbo_nodes);
    }

    int needed_floats = m_edges.size() * 2 * 3;
    float* vbo_edges_data = new float[needed_floats];
    float* dest = vbo_edges_data;
    memset(vbo_edges_data, 0, needed_floats * sizeof(float));
    for (int i = 0; i < m_edges.size(); ++i)
    {
        Node3D* node = m_edges[i]->u();
        *dest = node->x(); dest++;
        *dest = node->y(); dest++;
        *dest = node->z(); dest++;

        node = m_edges[i]->v();
        *dest = node->x(); dest++;
        *dest = node->y(); dest++;
        *dest = node->z(); dest++;
    }

    glGenBuffers(1, &m_vbo_edges);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_edges);
    glBufferData(GL_ARRAY_BUFFER, needed_floats * sizeof(float), vbo_edges_data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete [] vbo_edges_data;

    // Similar for nodes:
    // vec3 location
    // vec3 offset
    // vec3 color
    // vec2 selected_marked
    //
    // 6 vertices (2 triangles)
    needed_floats = m_nodes.size() * (3+3+3+2) * 6;
    float* vbo_nodes_data = new float[needed_floats];
    dest = vbo_nodes_data;
    memset(vbo_nodes_data, 0, needed_floats * sizeof(float));
    // 2D rectangle offsets (2 triangles)
    const float x_offsets[] = {-0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f};
    const float y_offsets[] = {-0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f};

    for (int i = 0; i < m_nodes.size(); ++i)
    {
        Node3D* node = m_nodes[i];
        QColor color = node->color();
        float x = node->x();
        float y = node->y();
        float z = node->z();
        float red = color.red() / 255.f;
        float green = color.green() / 255.f;
        float blue = color.blue() / 255.f;
        float selected = node->selected() ? 1.f : 0.f;
        float marked = node->marked() ? 1.f : 0.f;

        for (int v = 0; v < 6; ++v)
        {
            *dest = x; dest++;
            *dest = y; dest++;
            *dest = z; dest++;

            *dest = x_offsets[v]; dest++;
            *dest = y_offsets[v]; dest++;
            *dest = 0.f; dest++;

            *dest = red; dest++;
            *dest = green; dest++;
            *dest = blue; dest++;

            *dest = selected; dest++;
            *dest = marked; dest++;
        }
    }

    glGenBuffers(1, &m_vbo_nodes);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_nodes);
    glBufferData(GL_ARRAY_BUFFER, needed_floats * sizeof(float), vbo_nodes_data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete [] vbo_nodes_data;

    m_vbos_generated = true;
}

void Canvas3D::draw_edges()
{
    if (!m_vbos_generated)
        return;

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_edges);
    int vertex_size = 3;
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size*4, BUFFER_OFFSET(0));
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_LINES, 0, m_edges.size() * 2);
    glDisableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Canvas3D::draw_nodes()
{
    if (!m_vbos_generated)
        return;

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_nodes);
    int vertex_size = (3+3+3+2)*4;
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, BUFFER_OFFSET(0));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, BUFFER_OFFSET(3*4));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size, BUFFER_OFFSET(6*4));
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, vertex_size, BUFFER_OFFSET(9*4));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glDrawArrays(GL_TRIANGLES, 0, m_nodes.size() * 6);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

#include "canvas3d.moc"
