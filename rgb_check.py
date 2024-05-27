import pyrealsense2 as rs
import numpy as np
import cv2

# RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()

# 카메라 스트리밍 시작
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 클릭한 좌표의 HSV 값을 출력하는 함수
def print_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_value = hsv_image[y, x]
        print(f'Clicked HSV Value: H={hsv_value[0]}, S={hsv_value[1]}, V={hsv_value[2]}')

cv2.namedWindow('Filtered HSV Image')
cv2.setMouseCallback('Filtered HSV Image', print_hsv_value)

try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 이미지 데이터를 NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # BGR 이미지를 HSV 이미지로 변환
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # HSV 값 범위로 필터링
        lower_bound = np.array([175, 140, 120])
        upper_bound = np.array([180, 250, 250])

        # 마스크 생성
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # 마스크를 이용해 원본 이미지에서 필터링된 이미지 생성
        filtered_image = cv2.bitwise_and(color_image, color_image, mask=mask)

        # 결과 이미지 표시
        cv2.imshow('Filtered HSV Image', hsv_image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 스트리밍 중지 및 리소스 해제
    pipeline.stop()
    cv2.destroyAllWindows()
