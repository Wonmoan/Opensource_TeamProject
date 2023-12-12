
import cv2
import numpy as np
import time
import random
import os

# 저장할 경로
output_path = r"C:\Users\HWANG\PycharmProjects\pythonProject2\images"

# 디렉토리가 없다면 생성`
try:
    os.makedirs(output_path)
except FileExistsError:
    pass

# 카메라 및 파일 출력 관련 설정
cap = cv2.VideoCapture(0)  # 0은 기본 카메라, 만약 다른 카메라를 사용하려면 적절한 인덱스로 변경

current_time = time.time()
capture_count = 0  # 총 몇 장을 찍었는지를 저장하는 변수
rect_info = None   # 직사각형 정보를 저장하는 변수

while capture_count < 4:
    # 프레임 읽기
    ret, frame = cap.read()

    # 프레임 읽기에 실패하면 종료
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 5초에 한 번 실행
    if time.time() - current_time > 5:
        # 랜덤한 직사각형의 좌표 생성
        x, y, w, h = np.random.randint(0, frame.shape[1] - 300), np.random.randint(0, frame.shape[0] - 300), 300, 300

        # 랜덤한 직사각형 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 랜덤 직사각형 내부 영역 캡처
        captured_region = frame[y:y + h, x:x + w]

        # 이미지 파일로 저장
        image_name = f"captured_image_{capture_count + 1}.png"
        cv2.imwrite(os.path.join(output_path, image_name), captured_region)

        # 직사각형 정보 저장
        rect_info = (x, y, w, h)

        # 현재 시간 업데이트
        current_time = time.time()

        # 찍은 횟수 증가
        capture_count += 1

    # 이전에 그려진 직사각형이 있다면 계속해서 화면에 표시
    if rect_info is not None:
        x, y, w, h = rect_info
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 화면에 표시
    cv2.imshow("Random Rectangle Capture", frame)

    # 종료 키가 눌리면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()


