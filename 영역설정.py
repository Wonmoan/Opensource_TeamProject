import cv2
import mediapipe as mp

# 미디어 파이의 Hand 모듈을 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# OpenCV VideoCapture 객체를 생성하여 웹캠을 엽니다
cap     = cv2.VideoCapture(0)

# 화면을 8등분할하기 위한 좌표
regions = [
    (0, 0, 1/4, 1/2),   # 1번 영역
    (1/4, 0, 2/4, 1/2),  # 2번 영역
    (2/4, 0, 3/4, 1/2),  # 3번 영역
    (3/4, 0, 1, 1/2),    # 4번 영역
    (0, 1/2, 1/4, 1),    # 5번 영역
    (1/4, 1/2, 2/4, 1),  # 6번 영역
    (2/4, 1/2, 3/4, 1),  # 7번 영역
    (3/4, 1/2, 1, 1),    # 8번 영역
]

# 각 영역에 대응하는 문자
region_labels = ['1', '2', '3', '4', '5', '6', '7', '8']

# 이전에 그려진 선들을 저장하기 위한 리스트
drawn_lines = []

while cap.isOpened():
    # 웹캠으로부터 프레임을 읽어옵니다
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # 프레임을 RGB 포맷으로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 미디어 파이를 이용하여 손 감지 수행
    results = hands.process(rgb_frame)

    # 감지된 손의 위치가 존재하는지 확인하고 좌표 추출
    if results.multi_hand_landmarks:
        # 현재 프레임에서 그려질 선들을 저장하기 위한 리스트
        current_lines = []

        for hand_landmarks in results.multi_hand_landmarks:
            for region_id, (x1, y1, x2, y2) in enumerate(regions):
                # 영역의 좌표를 프레임 크기에 맞게 조정
                x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])

                # 손의 중간 지점을 계산
                cx, cy = int((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x + hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x) * frame.shape[1] / 2), int((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y + hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y) * frame.shape[0] / 2)

                # 손의 중간 지점이 영역 내에 있는지 확인하고 문자 및 영역 표시
                if x1 < cx < x2 and y1 < cy < y2:
                    cv2.putText(frame, region_labels[region_id], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # 현재 영역의 경계에 선을 그리기
                    cv2.line(frame, (x1, y1), (x1, y2), (0, 255, 0), 2)
                    cv2.line(frame, (x2, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.line(frame, (x1, y1), (x2, y1), (0, 255, 0), 2)
                    cv2.line(frame, (x1, y2), (x2, y2), (0, 255, 0), 2)

                    # 현재 그린 선을 리스트에 추가
                    current_lines.extend([(x1, y1, x1, y2), (x2, y1, x2, y2), (x1, y1, x2, y1), (x1, y2, x2, y2)])

        # 현재 그려진 선들을 전역 리스트에 추가
        drawn_lines.extend(current_lines)

        # 전역 리스트에 저장된 선들을 현재 프레임에 그리기
        for line in drawn_lines:
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("Hand Tracking", frame)

    # 종료 키 (q)를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()