#cython: embedsignature=True
#cython: language_level=3

import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def val_from_str_add_cont(self, s):
    if s in self.unknown_str:
        return self.Unknown
    val = float(s)
    cdef int i
    cdef int ndec
    cdef str s1
    cdef int nd
    cdef int ad = self.adjust_decimals
    if ad and isinstance(s, str):
        nd = self._number_of_decimals
        s1 = s.strip()
        i = s1.find(".")
        ndec = len(s1) - i - 1 if i > 0 else 0
        if ndec > nd or ad == 2:
            self.number_of_decimals = ndec
            self.adjust_decimals = 1
    return val
