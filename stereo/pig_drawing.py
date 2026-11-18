"""Outlines, body lines and weights over a frame, the same on the tab and in the marked-up video.

The marks are kept as polygons in the pixels of the analysed frame, so a video analysed reduced is
drawn back over its full-size frames. Text is Latin because OpenCV fonts have no Cyrillic.
"""
import cv2
import numpy as np

OUTLINE_OPACITY = .35
# Labels and lines grow with the picture: 1400 pixels of width per unit of font size.
LABEL_WIDTH_PER_UNIT = 1400
LINE_WIDTH_PER_PIXEL = 640
CONTOUR_EPSILON_PX = 1.
# Below this a weight is written with tenths: a piglet of 1.4 kg is not one of 0.6 kg.
TENTHS_BELOW_KG = 20
BONE = (255, 255, 255)
SHADOW = (20, 20, 20)


def outline_contours(mask) -> tuple:
    """The outer edge of an outline as polygons, a small fraction of the size of the mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return tuple(cv2.approxPolyDP(contour, CONTOUR_EPSILON_PX, True).astype(np.int32)
                 for contour in contours if len(contour) >= 3)


def kg_text(mass) -> str:
    return f"{mass:.1f}" if mass < TENTHS_BELOW_KG else f"{mass:.0f}"


def label_text(track_id, estimate) -> str:
    return f"#{track_id}" + ("" if estimate is None else f" {kg_text(estimate.mass_kg)} kg")


def skeleton(shape) -> np.ndarray:
    """The body line from the snout through the neck and along the spine to the rump."""
    return np.array((shape.head, *shape.spine), np.float64)


def girth_line(shape) -> np.ndarray:
    """Where the girth is taken: across the body behind the shoulders, as wide as the body there."""
    spine = np.array(shape.spine, np.float64)
    point = np.array(shape.girth_point, np.float64)
    nearest = int(np.argmin(np.linalg.norm(spine - point, axis=1)))
    ahead, behind = spine[max(0, nearest - 1)], spine[min(len(spine) - 1, nearest + 1)]
    along = behind - ahead
    length = float(np.linalg.norm(along))
    if length < 1e-6:
        return np.array((point, point))
    across = np.array((-along[1], along[0])) / length * shape.girth_px / 2
    return np.array((point - across, point + across))


def pixels(points, scale) -> np.ndarray:
    return np.round(np.asarray(points, np.float64) * scale).astype(np.int32)


def draw_marks(canvas, marks, texts, scale=1.) -> np.ndarray:
    """Draw every animal over the canvas in place and return it.

    Each mark has track_id, box, color, contours and shape (or None); `scale` takes the analysed
    frame to the canvas.
    """
    width = canvas.shape[1]
    size = max(.5, width / LABEL_WIDTH_PER_UNIT)
    weight = 2 if size >= .8 else 1
    line = max(1, round(width / LINE_WIDTH_PER_PIXEL))
    outlines = [[pixels(contour, scale) for contour in mark.contours] for mark in marks]
    if any(outlines):
        tinted = canvas.copy()
        for mark, polygons in zip(marks, outlines):
            if polygons:
                cv2.fillPoly(tinted, polygons, mark.color)
        cv2.addWeighted(tinted, OUTLINE_OPACITY, canvas, 1 - OUTLINE_OPACITY, 0, dst=canvas)
        for mark, polygons in zip(marks, outlines):
            if polygons:
                cv2.polylines(canvas, polygons, True, mark.color, line, cv2.LINE_AA)
    for mark, text in zip(marks, texts):
        x, y = int(mark.box[0] * scale), int(mark.box[1] * scale) + 20
        if mark.shape is not None:
            bones = pixels(skeleton(mark.shape), scale)
            girth = pixels(girth_line(mark.shape), scale)
            for colour, extra in ((SHADOW, 2), (BONE, 0)):
                cv2.polylines(canvas, [bones], False, colour, 2 * line + extra, cv2.LINE_AA)
                cv2.line(canvas, tuple(girth[0]), tuple(girth[1]), colour, line + extra, cv2.LINE_AA)
            for joint in bones[1:]:
                cv2.circle(canvas, tuple(joint), 2 * line + 1, mark.color, -1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(joint), 2 * line + 1, SHADOW, 1, cv2.LINE_AA)
            neck, rump = bones[1], bones[-1]
            x, y = (int(neck[0]) + int(rump[0])) // 2, (int(neck[1]) + int(rump[1])) // 2
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, weight)
        left = max(0, min(width - text_width - 8, x - text_width // 2 - 4))
        top = max(0, min(canvas.shape[0] - text_height - 8, y - text_height - 6))
        cv2.rectangle(canvas, (left, top), (left + text_width + 8, top + text_height + 8), mark.color, -1)
        cv2.putText(canvas, text, (left + 4, top + text_height + 3), cv2.FONT_HERSHEY_SIMPLEX, size, SHADOW, weight,
                    cv2.LINE_AA)
    return canvas
