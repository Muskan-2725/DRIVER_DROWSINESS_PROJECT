


def calculate_mar(face_landmarks, frame):
    h,w,_ = frame.shape

    # mouth landmark points
    top = face_landmarks.landmark[13]
    bottom = face_landmarks.landmark[14]
    left = face_landmarks.landmark[78]
    right = face_landmarks.landmark[308]


    # Convert to pixels
    top_pt = (int(top.x*w), int(top.y*h))
    bottom_pt = (int(bottom.x*w), int(bottom.y*h))

    vertical = abs(top_pt[1] - bottom_pt[1])
    horizontal = abs(int(left.x*w) - int(right.x*w))

    mar = vertical / horizontal

    return mar , [top_pt, bottom_pt]