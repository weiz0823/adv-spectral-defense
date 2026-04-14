def load_pascal_ann(name: str):
    """Load PASCAL annotation file.

    Returns:
        xyxy format bounding boxes.
    """
    bboxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []
    with open(name, "r", encoding="latin-1") as f:
        for line in f:
            if line.startswith("Image size (X x Y x C) :"):
                words = line.split()
                iw = int(words[8])
                ih = int(words[10])
            if line.startswith("Bounding box for object"):
                words = line.split()
                xmin = int(words[12][1:-1])
                ymin = int(words[13][:-1])
                xmax = int(words[15][1:-1])
                ymax = int(words[16][:-1])
                # x = (xmax + xmin) / 2
                # y = (ymax + ymin) / 2
                # width = xmax - xmin
                # heigth = ymax - ymin
                bboxes.append((xmin / iw, ymin / ih, xmax / iw, ymax / ih))
                # Currently we only deal with INRIA which only contains person.
                labels.append(0)
    return bboxes, labels
