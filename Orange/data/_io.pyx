from libc.stdio cimport *
from libc.string cimport *

cdef extern from "stdio.h" nogil:
    int fgetc(FILE *STREAM)
    int ungetc(char c, FILE *STREAM)

cdef enum State:
    BEGIN_LINE, READ_START_ATOM, READ, QUOTED, END_QUOTED,
    COMMENT, CARRIAGE_RETURNED, ESCAPE, EQUALS, END_LINE

def prescan(fname):
    cdef:
        State state
        char c
        int ci
        char *not_in_atom = "#=,|;\t\n\r\x00"
        int col, line
        int in_line
        char atom[10240], *atomp, *atome, *endc
        int n_attributes = 0
        int n_classes = 0
        int n_metas = 0
        int n_lines = 0
        int *output_count
        char f_eof = 0

    cdef FILE *f = fopen(fname, "rb")
    if f == NULL:
        raise IOError("File '{}' cannot be opened".format(fname))

    atome = atom + 10240
    attributes = set()
    classes = set()
    metas = set()

    try:
        state = BEGIN_LINE
        while True:
            if state == BEGIN_LINE:
                output = attributes
                output_count = &n_attributes
                col = in_line = 0
                line += 1
                state = READ_START_ATOM

            if state != END_LINE:
                ci = fgetc(f)
                if ci == EOF:
                    f_eof = 1
                    c = "\x00"
                else:
                    c = <char>ci
                col += 1
                if col > 100:
                    return

            if state == READ_START_ATOM:
                atomp = atom
                if c == "," or c == " " or c == "\t":
                    continue
                elif c == '"':
                    state = QUOTED
                    continue
                # fallthrough
                state = READ

            if state == ESCAPE:
                if c == "t":    c = "\t"
                elif c == "n":    c = "\n"
                elif c == "r":    c = "\r"
                elif c == '"' or c == "'" or c == "\\" or c == " ":
                    pass
                elif c == "\r" or c == "\n":
                    raise ValueError("{}:{}:{}: end of line in escape sequence"
                        .format(fname, line, col))
                elif f_eof:
                    raise ValueError("{}:{}:{}: end of file in escape sequence"
                        .format(fname, line, col))
                else:
                    raise ValueError("{}:{}:{}: unrecognized escape sequence"
                        .format(fname, line, col))
                # fallthrough
                state = READ

            if state == READ or state == EQUALS:
                endc = strchr(not_in_atom, c)
                if endc == NULL:
                    if state == READ:
                        atomp[0] = c
                        atomp += 1
                        if atomp == atome:
                            raise ValueError("{}:{}:{}: value name too long"
                                .format(fname, line, col))
                else:
                    if state == READ and atomp != atom:
                        # strip whitespace on the right
                        while atomp != atom and (
                                atomp[-1] == " " or atomp[-1] == "\t"):
                            atomp -= 1
                        if atomp != atom:
                            atomp[0] = "\x00"
                            output.add(atom)
                            output_count[0] += 1
                            in_line += 1
                    if c == "|":
                        if output_count != &n_attributes:
                            raise ValueError(
                                "{}:{}:{}: classes should follow attributes"
                                .format(fname, line, col))
                        output = classes
                        output_count = &n_classes
                        state = READ_START_ATOM
                    elif c == ";":
                        if output_count == &n_metas:
                            raise ValueError("{}:{}:{} duplicated semi-colons"
                                .format(fname, line, col))
                        output = metas
                        output_count = &n_metas
                        state = READ_START_ATOM
                    elif c == "," or c == "\t":
                        state = READ_START_ATOM
                    elif c == "#":
                        state = COMMENT
                    elif c == "\\":
                        state = ESCAPE
                    elif c == "\n" or f_eof:
                        state = END_LINE
                    elif c == "\r":
                        state = CARRIAGE_RETURNED
                    elif c == "=":
                        if state == EQUALS:
                            raise ValueError("{}:{}:{}: invalid value"
                                .format(fname, line, col))
                        state = EQUALS

            elif state == QUOTED:
                if c == "\r" or c == "\n":
                    raise ValueError(
                        "{}:{}:{}: end of line within a quoted value"
                        .format(fname, line, col))
                elif f_eof:
                    raise ValueError(
                        "{}:{}:{}: end of file within a quoted value"
                        .format(fname, line, col))
                elif c == '"':
                    state = END_QUOTED
                else:
                    atomp[0] = c
                    atomp += 1
                    if atomp == atome:
                        raise ValueError("{}:{}:{}: value name too long"
                            .format(fname, line, col))

            elif state == END_QUOTED:
                endc = strchr(not_in_atom, c)
                if endc == NULL:
                    raise ValueError("{}:{}:{}: quoted value should be "
                        "followed by value separator or end of line"
                        .format(fname, line, col))
                if atomp != atom:
                    atomp = "\x00"
                    output.add(atom)
                    output_count[0] += 1
                    in_line += 1
                    if c == "#":
                        state = COMMENT
                    elif c == "\n" or f_eof:
                        state = END_LINE
                    elif c == "\r":
                        state = CARRIAGE_RETURNED
                    else:
                        state = READ_START_ATOM

            elif state == COMMENT:
                if c == "\n" or f_eof:
                    state = END_LINE
                elif c == "\r":
                    state = CARRIAGE_RETURNED

            elif state == CARRIAGE_RETURNED:
                if c != "\n" and not f_eof:
                    ungetc(c, f)
                state = END_LINE

            elif state == END_LINE:
                if in_line:
                    n_lines += 1
                if f_eof:
                    return (attributes, classes, metas,
                            n_attributes, n_classes, n_metas, n_lines)
                state = BEGIN_LINE
    finally:
        fclose(f)