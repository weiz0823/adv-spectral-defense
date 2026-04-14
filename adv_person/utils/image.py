from .typing import _size_2_t, _pair

CM_PER_INCH = 2.54


def compute_image_shape(
    inch: _size_2_t, dpi: _size_2_t, pixel_ori: _size_2_t = None, is_cm=False
):
    h_ori, w_ori = _pair(pixel_ori)
    h_inch, w_inch = _pair(inch)
    dpi = _pair(dpi)
    if w_inch is None:
        w_inch = w_ori / h_ori * h_inch
    if h_inch is None:
        h_inch = h_ori / w_ori * w_inch
    if is_cm:
        w_inch = w_inch / CM_PER_INCH
        h_inch = h_inch / CM_PER_INCH
    return int(h_inch * dpi[0]), int(w_inch * dpi[1])
