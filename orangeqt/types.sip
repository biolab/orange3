// QMap<int, TYPE*> is implemented as a Python dictionary.
template<TYPE>
%MappedType QMap<int, TYPE*> /DocType="dict-of-int-TYPE"/
{
%TypeHeaderCode
#include <qmap.h>
%End

%ConvertFromTypeCode
    // Create the dictionary.
    PyObject *d = PyDict_New();

    if (!d)
        return NULL;

    // Set the dictionary elements.
    QMap<int, TYPE*>::const_iterator i = sipCpp->constBegin();

    while (i != sipCpp->constEnd())
    {
        TYPE *t = i.value();

        PyObject *kobj = SIPLong_FromLong(i.key());
        PyObject *tobj = sipConvertFromType(t, sipType_TYPE, sipTransferObj);

        if (kobj == NULL || tobj == NULL || PyDict_SetItem(d, kobj, tobj) < 0)
        {
            Py_DECREF(d);

            if (kobj)
            {
                Py_DECREF(kobj);
            }

            if (tobj)
            {
                Py_DECREF(tobj);
            }
            else
            {
                delete t;
            }

            return NULL;
        }

        Py_DECREF(kobj);
        Py_DECREF(tobj);

        ++i;
    }

    return d;
%End

%ConvertToTypeCode
    PyObject *kobj, *tobj;
    SIP_SSIZE_T i = 0;

    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
    {
        if (!PyDict_Check(sipPy))
            return 0;

        while (PyDict_Next(sipPy, &i, &kobj, &tobj))
            if (!sipCanConvertToType(tobj, sipType_TYPE, SIP_NOT_NONE))
                return 0;

        return 1;
    }

    QMap<int, TYPE*> *qm = new QMap<int, TYPE*>;
 
    while (PyDict_Next(sipPy, &i, &kobj, &tobj))
    {
        int state, k = SIPLong_AsLong(kobj);
        TYPE *t = reinterpret_cast<TYPE *>(sipConvertToType(tobj, sipType_TYPE, sipTransferObj, SIP_NOT_NONE, &state, sipIsErr));
 
        if (*sipIsErr)
        {
            sipReleaseType(t, sipType_TYPE, state);

            delete qm;
            return 0;
        }

        qm->insert(k, t);

        sipReleaseType(t, sipType_TYPE, state);
    }
 
    *sipCppPtr = qm;
 
    return sipGetState(sipTransferObj);
%End
};

// QMap<int, bool> is implemented as a Python dictionary.
%MappedType QMap<int, bool> /DocType="dict-of-int-bool"/
{
%TypeHeaderCode
#include <qmap.h>
%End

%ConvertFromTypeCode
    // Create the dictionary.
    PyObject *d = PyDict_New();

    if (!d)
        return NULL;

    // Set the dictionary elements.
    QMap<int, bool>::const_iterator i = sipCpp->constBegin();

    while (i != sipCpp->constEnd())
    {
        PyObject *kobj = SIPLong_FromLong(i.key());
        PyObject *tobj = Py_False;
        
        if (i.value())
        {
        	tobj = Py_True;
        }

        if (kobj == NULL || tobj == NULL || PyDict_SetItem(d, kobj, tobj) < 0)
        {
            Py_DECREF(d);
            if (kobj)
            {
                Py_DECREF(kobj);
            }
            if (tobj)
            {
                Py_DECREF(tobj);
            }
            return NULL;
        }

        Py_DECREF(kobj);
        Py_DECREF(tobj);

        ++i;
    }

    return d;
%End

%ConvertToTypeCode
    PyObject *kobj, *tobj;
    SIP_SSIZE_T i = 0;

    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
    {
        if (!PyDict_Check(sipPy))
            return 0;
        return 1;
    }

    QMap<int, bool> *qm = new QMap<int, bool>;
 
    while (PyDict_Next(sipPy, &i, &kobj, &tobj))
    {
        int k = SIPLong_AsLong(kobj);
        bool t = false;
        if (tobj == Py_True)
        {
        	t = true;
        }
 
        if (*sipIsErr)
        {
            delete qm;
            return 0;
        }

        qm->insert(k, t);
    }
 
    *sipCppPtr = qm;
 
    return sipGetState(sipTransferObj);
%End
};

// QMap<int, double> is implemented as a Python dictionary.
%MappedType QMap<int, double> /DocType="dict-of-int-float"/
{
%TypeHeaderCode
#include <qmap.h>
%End

%ConvertFromTypeCode
    // Create the dictionary.
    PyObject *d = PyDict_New();

    if (!d)
        return NULL;

    // Set the dictionary elements.
    QMap<int, double>::const_iterator i = sipCpp->constBegin();

    while (i != sipCpp->constEnd())
    {
        PyObject *kobj = SIPLong_FromLong(i.key());
        PyObject *tobj = PyFloat_FromDouble(i.value());
        if (kobj == NULL || tobj == NULL || PyDict_SetItem(d, kobj, tobj) < 0)
        {
            Py_DECREF(d);
            if (kobj)
            {
                Py_DECREF(kobj);
            }
            if (tobj)
            {
                Py_DECREF(tobj);
            }
            return NULL;
        }

        Py_DECREF(kobj);
        Py_DECREF(tobj);

        ++i;
    }

    return d;
%End


%ConvertToTypeCode
    PyObject *kobj, *tobj;
    SIP_SSIZE_T i = 0;

    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
    {
        if (!PyDict_Check(sipPy))
            return 0;
        return 1;
    }

    QMap<int, double> *qm = new QMap<int, double>;
 
    while (PyDict_Next(sipPy, &i, &kobj, &tobj))
    {
        int k = SIPLong_AsLong(kobj);
        double t = PyFloat_AsDouble(tobj);
 
        if (*sipIsErr)
        {
            delete qm;
            return 0;
        }

        qm->insert(k, t);
    }
 
    *sipCppPtr = qm;
 
    return sipGetState(sipTransferObj);
%End
};

// QMap<int, QPair<double, double> > is implemented as a Python dictionary of tuples.
%MappedType QMap<int, QPair<double, double> > /DocType="dict-of-int-tuple"/
{
%TypeHeaderCode
#include <qmap.h>
%End

%ConvertFromTypeCode
    // Create the dictionary.
    PyObject *d = PyDict_New();

    if (!d)
        return NULL;

    // Set the dictionary elements.
    QMap<int, QPair<double, double> >::const_iterator i = sipCpp->constBegin();

    while (i != sipCpp->constEnd())
    {
        PyObject *kobj = SIPLong_FromLong(i.key());
        PyObject *tobj;
		if ((tobj = PyTuple_New(2)) == NULL)
			return NULL;
		
		PyTuple_SET_ITEM(tobj, 0, PyFloat_FromDouble((double)i.value().first));
		PyTuple_SET_ITEM(tobj, 1, PyFloat_FromDouble((double)i.value().second));
    
        if (kobj == NULL || tobj == NULL || PyDict_SetItem(d, kobj, tobj) < 0)
        {
            Py_DECREF(d);
            if (kobj)
            {
                Py_DECREF(kobj);
            }
            if (tobj)
            {
                Py_DECREF(tobj);
            }
            return NULL;
        }

        Py_DECREF(kobj);
        Py_DECREF(tobj);

        ++i;
    }

    return d;
%End


%ConvertToTypeCode
    PyObject *kobj, *tobj;
    SIP_SSIZE_T i = 0;

    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
    {
        if (!PyDict_Check(sipPy))
            return 0;
        return 1;
    }

    QMap<int, QPair<double, double> > *qm = new QMap<int, QPair<double, double> >;
 
    while (PyDict_Next(sipPy, &i, &kobj, &tobj))
    {
        int k = SIPLong_AsLong(kobj);
        double fst = PyFloat_AsDouble(PyTuple_GET_ITEM(tobj,0));
		double sec = PyFloat_AsDouble(PyTuple_GET_ITEM(tobj,1));
		
		QPair<double, double> t(fst,sec);
 
        if (*sipIsErr)
        {
            delete qm;
            return 0;
        }

        qm->insert(k, t);
    }
 
    *sipCppPtr = qm;
 
    return sipGetState(sipTransferObj);
%End
};


// QMap<int, Triple<double, double, double> > is implemented as a Python dictionary of 3-tuples.
%MappedType QMap<int, Triple<double, double, double> > /DocType="dict-of-int-tuple"/
{
%TypeHeaderCode
#include <qmap.h>
#include "triple.h"
%End

%ConvertFromTypeCode
    // Create the dictionary.
    PyObject *d = PyDict_New();

    if (!d)
        return NULL;

    // Set the dictionary elements.
    QMap<int, Triple<double, double, double> >::const_iterator i = sipCpp->constBegin();

    while (i != sipCpp->constEnd())
    {
        PyObject *kobj = SIPLong_FromLong(i.key());
        PyObject *tobj;
		if ((tobj = PyTuple_New(3)) == NULL)
			return NULL;
		
		PyTuple_SET_ITEM(tobj, 0, PyFloat_FromDouble((double)i.value().first));
		PyTuple_SET_ITEM(tobj, 1, PyFloat_FromDouble((double)i.value().second));
		PyTuple_SET_ITEM(tobj, 2, PyFloat_FromDouble((double)i.value().third));
    
        if (kobj == NULL || tobj == NULL || PyDict_SetItem(d, kobj, tobj) < 0)
        {
            Py_DECREF(d);
            if (kobj)
            {
                Py_DECREF(kobj);
            }
            if (tobj)
            {
                Py_DECREF(tobj);
            }
            return NULL;
        }

        Py_DECREF(kobj);
        Py_DECREF(tobj);

        ++i;
    }

    return d;
%End

%ConvertToTypeCode
    PyObject *kobj, *tobj;
    SIP_SSIZE_T i = 0;

    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
    {
        if (!PyDict_Check(sipPy))
            return 0;
        return 1;
    }

    QMap<int, Triple<double, double, double> > *qm = new QMap<int, Triple<double, double, double> >;
 
    while (PyDict_Next(sipPy, &i, &kobj, &tobj))
    {
        int k = SIPLong_AsLong(kobj);
        double fst = PyFloat_AsDouble(PyTuple_GET_ITEM(tobj, 0));
		double sec = PyFloat_AsDouble(PyTuple_GET_ITEM(tobj, 1));
		double thr = PyFloat_AsDouble(PyTuple_GET_ITEM(tobj, 2));
		
		Triple<double, double, double> t(fst, sec, thr);
 
        if (*sipIsErr)
        {
            delete qm;
            return 0;
        }

        qm->insert(k, t);
    }
 
    *sipCppPtr = qm;
 
    return sipGetState(sipTransferObj);
%End
};


// Triple<double, double, double>
%MappedType Triple<double, double, double> /DocType="triple"/
{
%TypeHeaderCode
#include "triple.h"
%End

%ConvertFromTypeCode
    PyObject *t = PyTuple_New(3);

    if (!t)
        return NULL;

    PyTuple_SET_ITEM(t, 0, PyFloat_FromDouble((double)(sipCpp->first)));
    PyTuple_SET_ITEM(t, 1, PyFloat_FromDouble((double)(sipCpp->second)));
    PyTuple_SET_ITEM(t, 2, PyFloat_FromDouble((double)(sipCpp->third)));

    // TODO: any DECREF needed?

    return t;
%End

%ConvertToTypeCode
    if (sipIsErr == NULL)
    {
        // TODO
        return 1;
    }

    Triple<double, double, double> *tr = new Triple<double, double, double>(
        PyFloat_AsDouble(PyTuple_GET_ITEM(sipPy, 0)),
        PyFloat_AsDouble(PyTuple_GET_ITEM(sipPy, 1)),
        PyFloat_AsDouble(PyTuple_GET_ITEM(sipPy, 2)));
 
    if (*sipIsErr)
    {
        delete tr;
        return 0;
    }

    *sipCppPtr = tr;
 
    return sipGetState(sipTransferObj);
%End
};


%MappedType QList<bool> /DocType="list-of-int"/
{
%TypeHeaderCode
#include <qlist.h>
%End

%ConvertFromTypeCode
    // Create the list.
    PyObject *l;

    if ((l = PyList_New(sipCpp->size())) == NULL)
        return NULL;

    // Set the list elements.
    for (int i = 0; i < sipCpp->size(); ++i)
    {
        PyObject *pobj = sipCpp->value(i) ? Py_True : Py_False;

        if (pobj == NULL)
        {
            Py_DECREF(l);

            return NULL;
        }

        PyList_SET_ITEM(l, i, pobj);
    }

    return l;
%End

%ConvertToTypeCode
    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
        return (PySequence_Check(sipPy) && PySequence_Size(sipPy) >= 0);

    QList<bool> *ql = new QList<bool>;
    SIP_SSIZE_T len = PySequence_Size(sipPy);
 
    for (SIP_SSIZE_T i = 0; i < len; ++i)
    {
        PyObject *itm = PySequence_ITEM(sipPy, i);

        if (!itm)
        {
            delete ql;
            *sipIsErr = 1;

            return 0;
        }

        ql->append(itm == Py_True ? true : false);

        Py_DECREF(itm);
    }
 
    *sipCppPtr = ql;
 
    return sipGetState(sipTransferObj);
%End
};
