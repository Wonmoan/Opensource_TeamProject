## 8칸띄우기

## 사진 출력후 손가락 인식 창 새로 띄우기

##사진촬영후 촬영된 사진 이어붙이기 창에 띄우기
## 좌우반전과 사진 깔끔하게 저장하는 코드 합침
from PIL import Image
import shutil
import cv2
import mediapipe as mp
import numpy as np
import time
import random
import os
# 그리기 도구 지원해주는 서브 패키지
mp_drawing = mp.solutions.drawing_utils

# 손 감지 모듈
mp_hands = mp.solutions.hands

# 캠 키기
cap = cv2.VideoCapture(0)

# mp_hands의 Hands 정보를 설정하고 읽어들임
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    # 캠이 켜져있을때
    while cap.isOpened():

        # 캠 읽기 성공여부 success와 읽은 이미지를 image에 저장
        success, image = cap.read()

        # 캠 읽기 실패시 continue
        if not success:
            continue

        # 이미지 값 좌우반전 ( 캠 켰을때 반전된 이미지 보완 )
        # 이미지 값 순서를 BGR -> RGB로 변환
        # 이미지 순서가 RGB여야 Mediapipe 사용가능
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Image에서 손을 추적하고 결과를 result에 저장
        result = hands.process(image)

        # 이미지 값 순서를 RGB에서 BGR로 다시 바꿈
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 캠 화면에 띄울 텍스트 정의 ( 기본 값 )
        gesture_text = 'Cant found hand'

        # 결과 result가 제대로 추적이 되었을때
        if result.multi_hand_landmarks:

            # 첫 번째로 추적된 손을 hand_landmarks에 할당
            hand_landmarks = result.multi_hand_landmarks[0]

            # 검지 ~ 소지 까지의 다 펴져있는지에 대한 bool 변수들 선언
            finger_1 = False
            finger_2 = False
            finger_3 = False
            finger_4 = False
            finger_5 = False

            # 4번 마디가 2번 마디 보다 y값이 작으면 finger_1를 참
            if (hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y):
                finger_1 = True

            # 8번 마디가 6번 마디 보다 y값이 작으면 finger_2를 참
            if (hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y):
                finger_2 = True

            # 12번 마디가 10번 마디 보다 y값이 작으면 finger_3를 참
            if (hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y):
                finger_3 = True

            # 16번 마디가 14번 마디 보다 y값이 작으면 finger_4를 참
            if (hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y):
                finger_4 = True

            # 20번 마디가 18번 마디 보다 y값이 작으면 finger_5를 참
            if (hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y):
                finger_5 = True


            # 5손가락 다 펴져있으면 " 5번째 사진 선정 "
            if (finger_1 and finger_2 and finger_3 and finger_4 and finger_5):
                gesture_text = '5th select'
            elif (finger_2 and finger_3 and finger_4 and finger_5):
                gesture_text = '4th select'
            # 2,3,4,번째 손가락 펴져있으면 "3번째 사진 선정"
            elif(finger_2 and finger_3 and finger_4):
                gesture_text = '3th select'
            elif (finger_2 and finger_3 ):
                gesture_text = '2th select'
            # " 2번이 펴지면 2번째 사진 선정"
            elif (finger_2):
                gesture_text = '1th select'

            # 모든 손가락이 안펴져있으면 " 사진 촬영 준비중 "
            elif ((not finger_2) and (not finger_3) and (not finger_4)
                  and (not finger_5)):
                gesture_text = 'Photo Ready!!'
            elif ( (finger_5)):
                gesture_text = 'Photo Shoot!!'

            # 모두 아닐시 모르겠다는 텍스트
            else:
                gesture_text = 'Mo Ru Get Saw Yo'

            # 캠 화면에 손가락을 그림
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 캠화면에 텍스트를 작성
        cv2.putText(image, text='Hand shape : {}'.format(gesture_text)
                    , org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)





        # 캠 화면 ( 이미지 )을 화면에 띄움
        cv2.imshow('image', image)

        if (gesture_text == 'Photo Shoot!!'):
            cv2.destroyWindow('image')

            # 저장할 경로
            output_path = r"C:\Users\200718\PycharmProjects\pythonProject6\Images"

            # 디렉토리가 없다면 생성`
            try:
                os.makedirs(output_path)
            except FileExistsError:
                pass



            current_time = time.time()
            capture_count = 0  # 총 몇 장을 찍었는지를 저장하는 변수
            rect_info = None  # 직사각형 정보를 저장하는 변수

            while capture_count < 9:
                # 프레임 읽기
                ret, frame = cap.read()

                # 프레임 읽기에 실패하면 종료
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    break
                frame = cv2.flip(frame, 1)

                # 텍스트 추가
                gesture_text = 'Success'  # 텍스트 정의 (기본값)

                # 캠 화면에 텍스트 작성
                cv2.putText(frame, text='Take Photo!! : {}'.format(gesture_text),
                            org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 255), thickness=2)



                # 5초에 한 번 실행
                if time.time() - current_time > 5:
                    # 랜덤한 직사각형의 좌표 생성
                    x, y, w, h = np.random.randint(0, frame.shape[1] - 300), np.random.randint(0, frame.shape[
                        0] - 300), 300, 300

                    # 랜덤한 직사각형 그리기
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)



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

                    # 랜덤 직사각형 내부 영역 캡처
                    captured_region = frame[y:y + h, x:x + w]

                    # 좌우 반전 적용
                    captured_region = cv2.flip(captured_region, 1)
                    if (capture_count == 9):
                        break;
                    image_name = f"captured_image_{capture_count}.png"
                    cv2.imwrite(os.path.join(output_path, image_name), captured_region)
                    # 이미지 파일로 저장

                    # 화면에 표시
                cv2.imshow("Random Rectangle Capture", frame)

                # 종료 키가 눌리면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 리소스 해제
            cap.release()
            cv2.destroyAllWindows()

            # 촬영된 사진들을 모두 보여주는 코드
            # 이미지를 불러올 경로
            image_folder = r"C:\Users\200718\PycharmProjects\pythonProject6\Images"

            # 모든 사진 파일을 담을 리스트
            image_files = []
            # 이미지 경로에서 파일을 리스트에 추가
            for file in os.listdir(image_folder):
                if file.endswith(".png"):
                    image_files.append(os.path.join(image_folder, file))

            # 이미지를 행렬로 읽어옴
            images = [cv2.imread(image_path) for image_path in image_files if cv2.imread(image_path) is not None]

            # 이미지들을 모두 같은 크기로 조정
            max_height = max(image.shape[0] for image in images)
            max_width = max(image.shape[1] for image in images)
            resized_images = [cv2.resize(image, (max_width, max_height)) for image in images]

            # 이미지들을 수평으로 연결하여 하나의 이미지로 만듦
            total_images = len(resized_images)
            images_per_row = 4  # 한 행에 표시할 이미지 수
            rows = int(np.ceil(total_images / images_per_row))

            # 각 행마다 이미지를 연결하여 하나의 이미지로 생성
            combined_rows = []
            for i in range(rows):
                start_index = i * images_per_row
                end_index = min((i + 1) * images_per_row, total_images)
                combined_row = np.hstack(resized_images[start_index:end_index])
                combined_rows.append(combined_row)

            # 모든 행을 수직으로 연결하여 하나의 이미지로 만듦
            combined_image = np.vstack(combined_rows)

            # 이미지 크기를 화면에 맞게 조절
            scale_percent = 100  # 초기 스케일 비율
            while combined_image.shape[1] > 800 or combined_image.shape[0] > 2550:  # 이미지가 화면 크기보다 크면 크기 조절
                scale_percent -= 5
                width = int(combined_image.shape[1] * scale_percent / 100)
                height = int(combined_image.shape[0] * scale_percent / 100)
                dim = (width, height)
                combined_image = cv2.resize(combined_image, dim, interpolation=cv2.INTER_AREA)

            # 결과 이미지 저장할 경로와 파일명 설정
            save_path = r"C:\Users\200718\PycharmProjects\pythonProject6\simages\allphoto.png"

            # 이미지 저장
            cv2.imwrite(save_path, combined_image)

            # 이미지를 창에 표시
            cv2.imshow('Combined Images', combined_image)
            # 창이 열려있는 동안 기다림
            cv2.waitKey(0)
            # 창 닫기
            cv2.destroyAllWindows()

            # 모든 창 닫기
            cv2.destroyAllWindows()

            # q입력시 종료
            if cv2.waitKey(1) == ord('q'):
                break

        # q입력시 종료
        if cv2.waitKey(1) == ord('q'):
            break

# 캠 종료
cap.release()
cv2.destroyAllWindows()

# 이전에 저장한 이미지 파일을 불러옴
saved_image = cv2.imread(save_path)
# 새로운 창에 이미지 표시
cv2.imshow('New Window', saved_image)

# 캠 키기
cap = cv2.VideoCapture(0)

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

selected_regions = set()
# 파일을 저장할 경로
destination_folder = r"C:\Users\200718\PycharmProjects\pythonProject6\Images\fimage"

# Images 폴더의 경로
images_folder = r"C:\Users\200718\PycharmProjects\pythonProject6\Images"
while cap.isOpened():
    # 웹캠으로부터 프레임을 읽어옵니다
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
        # 좌우 반전 적용
    frame = cv2.flip(frame, 1)  # 두 번째 매개변수가 0이면 상하반전, 1이면 좌우반전
    # 프레임을 RGB 포맷으로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 미디어 파이를 이용하여 손 감지 수행
    results = hands.process(rgb_frame)

    # 감지된 손의 위치가 존재하는지 확인하고 좌표 추출
    if results.multi_hand_landmarks:
        # 현재 프레임에서 그려질 선들을 저장하기 위한 리스트
        current_lines = []

        for hand_landmarks in results.multi_hand_landmarks:
            for region_id, (x1, y1, x2, y2) in enumerate(regions,start=1):
                # 영역의 좌표를 프레임 크기에 맞게 조정
                x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])

                # 손의 중간 지점을 계산
                cx, cy = int((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x + hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x) * frame.shape[1] / 2), int((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y + hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y) * frame.shape[0] / 2)

                # 손의 중간 지점이 영역 내에 있는지 확인하고 문자 및 영역 표시
                if x1 < cx < x2 and y1 < cy < y2:
                    selected_regions.add(region_id)  # 선택된 영역을 집합에 추가
                    cv2.putText(frame, region_labels[region_id-1], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

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


    # 선택된 각 영역에 해당하는 파일을 fimage에 복사
    for region_id in selected_regions:
        file_name = f"captured_image_{region_id}.png"
        file_path = os.path.join(images_folder, file_name)
        if os.path.exists(file_path):
            shutil.copyfile(file_path, os.path.join(destination_folder, file_name))

    # 4가지 영역이 선택되면 종료
    if len(selected_regions) >= 4:
        if len(current_lines) == 4:
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# 자원 해제
cap.release()
cv2.destroyAllWindows()
# 캠 종료
cap.release()
cv2.destroyAllWindows()


#인생네컷 출력

# 배경 사진 경로 (PNG 형식으로 변경)
background_path = r"C:\Users\200718\PycharmProjects\pythonProject6\Images\fimage\black.png"
background_png_path = r"C:\Users\200718\PycharmProjects\pythonProject6\Images\fimage\background.png"

background = Image.open(background_path)
background.save(background_png_path, format='PNG')

# 네 장의 사진 경로
photo_paths = []

for region_id in selected_regions:
    file_name = f"captured_image_{region_id}.png"
    file_path = os.path.join(image_folder, file_name)
    photo_paths.append(file_path)

# 배경 이미지 열기
background = Image.open(background_png_path).convert("RGBA")

# 배경 이미지 크기
bg_width, bg_height = background.size

# 간격 및 사진 크기 설정
gap = 45  # 조절 가능한 간격 (더 큰 값으로 조절)
photo_width, photo_height = 480, 200

# 사진을 세로로 배치하기 위한 초기 y 위치 계산
total_height = len(photo_paths) * (photo_height + gap) - gap
# Decrease the y_position increment for a higher placement of photos
y_position = (bg_height - total_height) // 4

# 네 장의 사진을 반복하며 배경에 삽입
for photo_path in photo_paths:
    photo = Image.open(photo_path).convert("RGBA")

    # 사진 크기 조절 (원하는 크기로 조절)
    photo = photo.resize((photo_width, photo_height))

    # 배경에 사진 삽입 (가운데 정렬하여 세로로 배치)
    x_position = (bg_width - photo_width) // 2
    background.paste(photo, (x_position, y_position), photo)

    # 조금씩 올리기 위해 y_position을 적게 증가
    y_position += int(photo_height * 0.8) + gap  # 더 많은 간격을 추가

# 결과 이미지 저장
background.save(r"C:\Users\200718\PycharmProjects\pythonProject6\Images\fimage\result.png")

# 화면에 출력
background.show()