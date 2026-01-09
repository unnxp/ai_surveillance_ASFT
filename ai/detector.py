def detect_person(frame, model, conf=0.4):
    results = model(frame, conf=conf)[0]
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        score = box.conf[0].cpu().numpy()
        detections.append([x1, y1, x2, y2, score])

    return detections
