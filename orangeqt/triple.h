#ifndef TRIPLE_H
#define TRIPLE_H

template <class X, class Y, class Z>
class Triple
{
public:
    X first;
    Y second;
    Z third;

    Triple(): first(X(0)), second(Y(0)), third(Z(0)) {}
    Triple(const X& f, const Y& s, const Z& t): first(f), second(s), third(t) {}
};

#endif
