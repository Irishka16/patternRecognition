import cv2
import numpy as np

# Завантаження чорно-білого зображення
image = cv2.imread('input_image.jpg', 0)  # Вкажіть шлях до вашого зображення

# Застосування глобальної бінаризації (поріг = 128)
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Знаходження контурів на бінарному зображенні
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Створення чорно-білої маски об'єкта
object_mask = np.zeros_like(image)

# Заповнення об'єкта білою фарбою на масці
cv2.drawContours(object_mask, contours, -1, (255), thickness=cv2.FILLED)

# Відображення та збереження маски об'єкта
cv2.imshow('Object Mask', object_mask)
cv2.imwrite('object_mask.jpg', object_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()

