def calculate_ear(face_landmarks, frame, return_separate=False):
    h, w, _ = frame.shape

    #  Left eye landmarks 
    l_top1 = face_landmarks.landmark[159]
    l_bottom1 = face_landmarks.landmark[145]
    l_top2 = face_landmarks.landmark[153]
    l_bottom2 = face_landmarks.landmark[144]
    l_left = face_landmarks.landmark[33]
    l_right = face_landmarks.landmark[133]

    #  Right eye landmarks 
    r_top1 = face_landmarks.landmark[386]
    r_bottom1 = face_landmarks.landmark[374]
    r_top2 = face_landmarks.landmark[382]
    r_bottom2 = face_landmarks.landmark[373]
    r_left = face_landmarks.landmark[362]
    r_right = face_landmarks.landmark[263]

    # Convert to pixels
    def pt(lm):
        return int(lm.x * w), int(lm.y * h)

    l_top_pt1, l_bottom_pt1 = pt(l_top1), pt(l_bottom1)
    l_top_pt2, l_bottom_pt2 = pt(l_top2), pt(l_bottom2)
    l_left_pt, l_right_pt = pt(l_left), pt(l_right)

    r_top_pt1, r_bottom_pt1 = pt(r_top1), pt(r_bottom1)
    r_top_pt2, r_bottom_pt2 = pt(r_top2), pt(r_bottom2)
    r_left_pt, r_right_pt = pt(r_left), pt(r_right)

    # EAR calculation 
    l_vert1 = abs(l_top_pt1[1] - l_bottom_pt1[1])
    l_vert2 = abs(l_top_pt2[1] - l_bottom_pt2[1])
    l_horizontal = abs(l_left_pt[0] - l_right_pt[0])
    l_ear = (l_vert1 + l_vert2) / (2 * l_horizontal)

    r_vert1 = abs(r_top_pt1[1] - r_bottom_pt1[1])
    r_vert2 = abs(r_top_pt2[1] - r_bottom_pt2[1])
    r_horizontal = abs(r_left_pt[0] - r_right_pt[0])
    r_ear = (r_vert1 + r_vert2) / (2 * r_horizontal)

    if return_separate:
        return l_ear, r_ear, [l_top_pt1, l_bottom_pt1, l_top_pt2, l_bottom_pt2, l_left_pt, l_right_pt], \
               [r_top_pt1, r_bottom_pt1, r_top_pt2, r_bottom_pt2, r_left_pt, r_right_pt]
    else:
        return (l_ear + r_ear)/2, [l_top_pt1, l_bottom_pt1, l_top_pt2, l_bottom_pt2,
                                   l_left_pt, l_right_pt, r_top_pt1, r_bottom_pt1,
                                   r_top_pt2, r_bottom_pt2, r_left_pt, r_right_pt]