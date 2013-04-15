class UnconnectedLinesCurve : Curve {

%TypeHeaderCode
#include "unconnectedlinescurve.h"
%End

public:
    UnconnectedLinesCurve(QGraphicsItem* parent /TransferThis/ = 0);
    virtual ~UnconnectedLinesCurve();
    
    virtual void update_properties();
};