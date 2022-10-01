    n_boxes = len(d['level'])
    confidences = d['conf']
    boxes = []
    for i in range(n_boxes):
        if int(float(confidences[i])) >= 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # print(area(w, h))
            if w*h < 100 or w*h > 6000: continue
            boxes.append([x, y, x + w, y + h])
            draw.rectangle([(x, y), (x + w, y + h)])