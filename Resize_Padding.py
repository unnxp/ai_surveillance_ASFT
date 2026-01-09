import cv2
def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)

    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (nw, nh))

    top = (new_shape[0] - nh) // 2
    bottom = new_shape[0] - nh - top
    left = (new_shape[1] - nw) // 2
    right = new_shape[1] - nw - left

    return cv2.copyMakeBorder(
        image_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

frame = letterbox(frame)
