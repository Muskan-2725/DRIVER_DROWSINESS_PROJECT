def get_face_direction(landmarks, frame):
    h, w, _ = frame.shape

    nose = landmarks.landmark[1]
    left_cheek = landmarks.landmark[234]
    right_cheek = landmarks.landmark[454]

    nose_x = int(nose.x * w)
    left_x = int(left_cheek.x * w)
    right_x = int(right_cheek.x * w)

    center = (left_x + right_x) // 2

    if nose_x < center - 20:
        return "LEFT"
    elif nose_x > center + 20:
        return "RIGHT"
    else:
        return "FORWARD"