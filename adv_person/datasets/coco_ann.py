def save_coco_ann(
    name: str, bboxes: list[tuple[float, float, float, float]], labels: list[int]
):
    with open(name, "w") as f:
        for box, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = box
            f.write(
                "{} {} {} {} {}\n".format(label, xmin, ymin, xmax - xmin, ymax - ymin)
            )


def save_coco_cxcywh_ann(
    name: str,
    bboxes: list[tuple[float, float, float, float]],
    labels: list[int],
):
    with open(name, "w") as f:
        for box, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = box
            f.write(
                "{} {} {} {} {}\n".format(
                    label,
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                )
            )


def save_detection_yolo(
    name: str,
    bboxes: list[tuple[float, float, float, float]],
    scores: list[float] = None,
    labels: list[int] = None,
):
    with open(name, "w") as f:
        for i, box in enumerate(bboxes):
            xmin, ymin, xmax, ymax = box
            if labels is not None:
                f.write("{} ".format(labels[i]))
            f.write(
                "{} {} {} {}".format(
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                )
            )
            if scores is not None:
                f.write(" {}".format(scores[i]))
            f.write("\n")


def load_coco_ann(name: str):
    """Load COCO annotation file.

    Returns:
        xyxy format bounding boxes.
    """
    bboxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []
    with open(name, "r") as f:
        for line in f:
            fields = line.rstrip().split()
            labels.append(int(fields[0]))
            x, y, w, h = map(float, fields[1:])
            bboxes.append((x, y, x + w, y + h))
    return bboxes, labels


def load_coco_cxcywh_ann(name: str):
    """Load COCO annotation file.

    Returns:
        xyxy format bounding boxes.
    """
    bboxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []
    with open(name, "r") as f:
        for line in f:
            fields = line.rstrip().split()
            labels.append(int(fields[0]))
            cx, cy, w, h = map(float, fields[1:])
            w2 = w / 2.0
            h2 = h / 2.0
            bboxes.append((cx - w2, cy - h2, cx + w2, cy + h2))
    return bboxes, labels


def load_detection_yolo(name: str):
    """Load YOLO annotation file (cxcywh).

    Returns:
        xyxy format bounding boxes.
    """
    bboxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []
    scores: list[float] = []
    with open(name, "r") as f:
        for line in f:
            fields = line.rstrip().split()
            labels.append(int(fields[0]))
            cx, cy, w, h = map(float, fields[1:5])
            w2 = w / 2.0
            h2 = h / 2.0
            bboxes.append((cx - w2, cy - h2, cx + w2, cy + h2))
            if len(fields) > 5:
                score = float(fields[5])
            else:
                score = 1.0
            scores.append(score)
    return bboxes, labels, scores
