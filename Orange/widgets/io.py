from orangewidget.io import (
    ImgFormat,
    PngFormat,
    SvgFormat,
    ClipboardFormat,
    PdfFormat,
    MatplotlibPDFFormat,
    MatplotlibFormat,
)

__all__ = [
    "ImgFormat", "PngFormat", "SvgFormat", "ClipboardFormat", "PdfFormat",
    "MatplotlibFormat", "MatplotlibPDFFormat",
]

# This is for backwards compatibility. This should never be imported from here.
FileFormat = ImgFormat
