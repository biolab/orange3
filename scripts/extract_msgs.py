import tokenize

def extract(source, _, comment_tags, options=None):
    for token in tokenize.tokenize(source.readline):
        if token.type == tokenize.STRING:
            msg = token.string
            if msg[0] == "f":
                msg = msg[1:]
            if msg[0] == msg[-1] == "'":
                msg = f'"{msg}"'
            yield token.start[0], "", msg, ""
