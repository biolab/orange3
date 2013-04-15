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


#include "multicurve.h"
#include "plot.h"

MultiCurve::MultiCurve(const QList< double >& x_data, const QList< double >& y_data): Curve()
{
    set_continuous(false);
    set_data(x_data, y_data);
}

MultiCurve::~MultiCurve()
{

}

void MultiCurve::set_point_colors(const QList< QColor >& colors)
{
    update_point_properties("color", colors);
}

void MultiCurve::set_point_labels(const QStringList& labels)
{
    update_point_properties("label", labels, false);
}

void MultiCurve::set_point_sizes(const QList<int>& sizes)
{
    update_point_properties("size", sizes, false);
}

void MultiCurve::set_point_symbols(const QList< int >& symbols)
{
    update_point_properties("symbol", symbols, false);
}

void MultiCurve::update_properties()
{
    update_point_coordinates();
}

void MultiCurve::shuffle_points()
{
    update_items(points(), PointShuffler(), UpdateContinuous);
}

void MultiCurve::set_alpha_value(int alpha)
{
    update_items(points(), PointAlphaUpdater(alpha), UpdateBrush);
}

void MultiCurve::set_points_marked(const QList< bool >& marked)
{
    update_point_properties("marked", marked, false);
}

