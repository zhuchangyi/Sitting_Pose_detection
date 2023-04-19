import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 19200)

cap.set(cv2.CAP_PROP_FPS, 1200)

frame_count = 0
while True:
    # 读取摄像头画面
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps, "FPS")
    print(frame.shape)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # # 将画面压缩到256*256分辨率
    # resized_frame = cv2.resize(frame, (256, 256))

    # 显示压缩后的画面
    cv2.imshow('Resized Frame', frame)

    # 等待退出
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
print("总共跑了 %d 帧" % frame_count)
cap.release()
cv2.destroyAllWindows()
