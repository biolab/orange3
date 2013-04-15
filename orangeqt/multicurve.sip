
class MultiCurve : Curve
{
%TypeHeaderCode
#include "multicurve.h"
%End
public:
    MultiCurve(const QList< double >& x_data, const QList< double >& y_data);
    virtual ~MultiCurve();
    
    void set_point_colors(const QList<QColor>& colors);
    void set_point_labels(const QStringList& colors);
    void set_point_symbols(const QList< int >& symbols);
    void set_point_sizes(const QList<int>& colors);

    void set_points_marked(const QList<bool>& marked);

    void shuffle_points();
    void set_alpha_value(int alpha);

    virtual void update_properties();
};
