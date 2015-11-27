from bs4 import BeautifulSoup
from docutils import nodes

import recommonmark.parser as parser
from recommonmark.parser import CommonMarkParser,\
    inline_html as old_inline_html

__all__ = ['CommonMarkParser']


def create_image_node(block):
    soup = BeautifulSoup(block.c, 'html.parser')
    images = soup.find_all('img')

    for img in images:
        img_node = nodes.image()
        img_node['uri'] = img.attrs.pop('src')
        for attr, value in img.attrs.items():
            img_node[attr] = value
        return img_node


def inline_html(block):
    return create_image_node(block) or old_inline_html(block)
parser.inline_html = inline_html
