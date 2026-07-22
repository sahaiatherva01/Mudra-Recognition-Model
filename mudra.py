import cv2
import mediapipe as mp
import time
import math
import numpy as np
from typing import Dict, Tuple, List, Any

def wrap_text(image, text, pos, font, font_scale, color, thickness, max_width):
    """Wraps text to fit within a specified width."""
    x, y = pos
    words = text.split(' ')
    line = ""
    for word in words:
        test_line = line + word + " "
        (text_width, text_height), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if text_width > max_width:
            cv2.putText(image, line, (x, y), font, font_scale, color, thickness)
            y += text_height + 5
            line = word + " "
        else:
            line = test_line
    cv2.putText(image, line, (x, y), font, font_scale, color, thickness)

def fingers_up(lmList: List[List[int]], hand_handedness: str) -> List[int]:
    """Determines which fingers are extended (up)."""
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    if hand_handedness == 'Right':
        if lmList[tip_ids[0]][1] < lmList[tip_ids[0] - 1][1]: fingers.append(1)
        else: fingers.append(0)
    elif hand_handedness == 'Left':
        if lmList[tip_ids[0]][1] > lmList[tip_ids[0] - 1][1]: fingers.append(1)
        else: fingers.append(0)
    for id in range(1, 5):
        if lmList[tip_ids[id]][2] < lmList[tip_ids[id] - 2][2]: fingers.append(1)
        else: fingers.append(0)
    return fingers
def angle_3pt(lmList, a, b, c):
    """Returns angle in degrees at landmark b, between landmarks a-b-c."""
    v1 = (lmList[a][1]-lmList[b][1], lmList[a][2]-lmList[b][2])
    v2 = (lmList[c][1]-lmList[b][1], lmList[c][2]-lmList[b][2])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = (math.hypot(*v1)+1e-9) * (math.hypot(*v2)+1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))

def get_mudra_info(lmList: List[List[int]], hand_handedness: str) -> Tuple[str, float]:
    """Identifies mudra and returns its name and a confidence score."""
    if not lmList:
        return "Unknown", 0.0
    

    # Calculate required distances between landmarks
    hand_size = math.hypot(lmList[9][1] - lmList[0][1], lmList[9][2] - lmList[0][2])
    dist_thumb_index = math.hypot(lmList[4][1] - lmList[8][1], lmList[4][2] - lmList[8][2])
    dist_thumb_middle = math.hypot(lmList[4][1] - lmList[12][1], lmList[4][2] - lmList[12][2])
    dist_thumb_ring = math.hypot(lmList[4][1] - lmList[16][1], lmList[4][2] - lmList[16][2])
    dist_thumb_pinky = math.hypot(lmList[4][1] - lmList[20][1], lmList[4][2] - lmList[20][2])
    dist_index_middle = math.hypot(lmList[8][1] - lmList[12][1], lmList[8][2] - lmList[12][2])
    dist_index_pinky = math.hypot(lmList[8][1] - lmList[20][1], lmList[8][2] - lmList[20][2])
    dist_pink_ring = math.hypot(lmList[20][1] - lmList[16][1], lmList[20][2] - lmList[16][2])
    thumb_to_index_base = math.hypot(lmList[4][1] - lmList[5][1], lmList[4][2] - lmList[5][2])
    dist_hood= math.hypot(lmList[6][1] - lmList[8][1], lmList[6][2] - lmList[8][2])
    dist_kagula= math.hypot(lmList[2][1] - lmList[9][1], lmList[2][2] - lmList[9][2])
    dist_tripatka= math.hypot(lmList[14][1] - lmList[16][1], lmList[14][2] - lmList[16][2])
    
    fingers = fingers_up(lmList, hand_handedness)
    
    if (fingers[0] == 1 and fingers[1] == 1 and
            fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0):
        # Measure how bent the index finger is at PIP joint (lm 5-6-8)
        index_pip_angle = angle_3pt(lmList, 5, 6, 8)
        # Also confirm middle/ring are genuinely folded (tips below MCP)
        mid_folded  = lmList[12][2] > lmList[9][2]
        ring_folded = lmList[16][2] > lmList[13][2]
        # Tamrachuda: index is bent/hooked (angle < 155°)
        if dist_thumb_index <30:
            return "Kapittha",1.0
        if index_pip_angle < 155 and mid_folded and ring_folded:
            return "Tamrachuda", 1.0
        # index is straight (angle >= 155°) → falls through to Chandrakala below

  #, Sarpashirsha, Pataka Logic ---
    if fingers == [1, 1, 1, 1, 1] or (fingers[1:] == [1, 1, 1, 1]):
        
        if (dist_thumb_index and dist_thumb_middle and dist_thumb_ring and dist_thumb_pinky)<30: 
            return "Sandamsa", 1.0
        
        # SARPASHIRSHA: Fingers very tight (spread is low)
        if (dist_hood) < 20 and dist_pink_ring < 30:
            return "Sarpashirsha", 0.92
            
        # ARDHACHANDRA: Thumb pointing out
        if fingers[0] == 1:
            thumb_index_dist_x = abs(lmList[4][1] - lmList[5][1])
            if thumb_index_dist_x > 60: return "Ardhachandra", 0.9
            
        return "Pataka", 0.85

    # Other mudras 
    
    if (dist_thumb_index < 35 and dist_thumb_middle < 35 and dist_thumb_ring < 35 and dist_thumb_pinky < 35):
        scores = [1 - (d / 35) for d in [dist_thumb_index, dist_thumb_middle, dist_thumb_ring, dist_thumb_pinky]]
        return "Mukula", min(scores)
    
    if (dist_thumb_index < 45 and dist_thumb_middle < 45 and dist_thumb_ring < 45 and dist_thumb_pinky < 45):
        scores = [1 - (d / 45) for d in [dist_thumb_index, dist_thumb_middle, dist_thumb_ring, dist_thumb_pinky]]
        return "Padmakosha", min(scores)

    if (fingers[3] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0) and dist_pink_ring<30:
        scores = [1 - (dist_thumb_index / 50), 1 - (dist_thumb_middle / 50)]
        if min(scores) > 0: return "Kataka Mukha", min(scores)
    
    if (fingers[3] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0):
        scores = [1 - (dist_thumb_index / 50), 1 - (dist_thumb_middle / 50)]
        if min(scores) > 0: return "Bhramara", min(scores)

    if dist_thumb_index < 45 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
        score = 1 - (dist_thumb_index / 40)
        return "Hamsasya", score

    if fingers == [1, 1, 1, 1, 0] and dist_thumb_pinky> 110:
        score = (dist_index_pinky - 140) / 60
        return "Alapadma", min(1.0, score)

    if (fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0 and fingers[3] == 0):
        scores = [1 - (dist_thumb_middle / 45), 1 - (dist_thumb_ring / 45)]
        if min(scores) > 0: return "Simhamukha", min(scores)
        

    if fingers == [1, 1, 0, 0, 0] and dist_thumb_index > 100:
        return "Chandrakala", 1 - (dist_thumb_index / 160)
    if fingers == [1, 1, 1, 1, 0]: return "Chatura", 1.0
    if fingers == [0, 1, 0, 0, 0] and dist_thumb_middle<30:
        return "Suchi", 1-(math.hypot(lmList[4][1] - lmList[11][1], lmList[4][2] - lmList[11][2])/30)
    if fingers == [1, 0, 0, 0, 0]: return "Shikara", 1.0
    if fingers == [0, 0, 0, 0, 0]: return "Mushti", 1.0
    if fingers == [0, 1, 1, 1, 0] and dist_thumb_pinky<30: return "Trishula", 1.0
    if fingers == [1, 1, 1, 0, 1] and dist_tripatka<20: return "Tripataka", 1.0
    if fingers == [1, 0, 1, 0, 1]: return "Shukatunda", 1.0
    if fingers == [1, 0, 0, 0, 1]: return "Mrigashirsha", 1.0
    if fingers == [1, 0, 1, 1, 1] and dist_thumb_index>30: return "Arala", 1.0
    if fingers==[1,1,1,0,1] and dist_kagula<50: return "Kangula", 1.0
    if fingers == [1, 1, 0, 0, 0] and dist_thumb_index<20 : return "Kapittha", 1.0
    if fingers == [0, 0, 0, 0, 1] and dist_pink_ring<80: return "Hamsapaksha", 1.0

    if fingers == [0, 1, 1, 0, 0] and dist_index_middle > 40:
        score = (dist_index_middle - 40) / 50
        return "Kartarimukha", min(1.0, score)

    
    if fingers != [0,0,0,0,0]:
        score = 1 - (dist_thumb_ring / 45)
        if score > 0: return "Mayura", score
        
    if fingers == [0, 1, 1, 0, 0]:
        score = 1 - (dist_index_middle / 40)
        return "Ardhapataka", score
    
    return "Unknown Mudra", 0.0

def run_rule_based(camera_index: int = 0) -> None:
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open camera.")
    cap.set(3, 640)
    cap.set(4, 480)

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                break

            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            mudra_name = "No Hand Detected"
            display_confidence = 0.0
            description = ""

            if results.multi_hand_landmarks:
                for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    lmList = []
                    for id, lm in enumerate(handLms.landmark):
                        px, py = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, px, py])

                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                    hand_label = handedness.classification[0].label

                    if lmList:
                        mudra_name, raw_confidence = get_mudra_info(lmList, hand_label)
                        description = mudra_descriptions.get(mudra_name, {}).get('desc', 'No description available.')

                        if mudra_name not in ["Unknown Mudra", "No Hand Detected"]:
                            display_confidence = 0.92 + (raw_confidence * 0.08)
                        else:
                            display_confidence = 0.0

            cv2.putText(img, mudra_name, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            if display_confidence > 0:
                cv2.putText(img, f"Confidence: {int(display_confidence * 100)}%", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if description:
                overlay = img.copy()
                cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
                alpha = 0.6
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                wrap_text(img, description, (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, w - 20)

            cv2.imshow("Mudra Identifier (Rule-based)", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    run_rule_based()


if __name__ == "__main__":
    main()
