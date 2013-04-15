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

#ifndef UNCONNECTEDLINESCURVE_H
#define UNCONNECTEDLINESCURVE_H

#include "curve.h"

class UnconnectedLinesCurve : public Curve
{
    Q_OBJECT
    
public:
    UnconnectedLinesCurve(QGraphicsItem* parent = 0);
    virtual ~UnconnectedLinesCurve();
    
    virtual void update_properties();    
    
private:    
    QGraphicsPathItem* m_path_item;
    QFutureWatcher< QPainterPath >* m_path_watcher;
    
public slots:
    void path_calculated();
};

#endif // UNCONNECTEDLINESCURVE_H
