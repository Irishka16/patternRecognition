import cv2
import numpy as np

# Зчитування зображення
image_path = "pear.jpg"
original_image = cv2.imread(image_path)

# Конвертування в ч/б варіант
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Глобальна бінаризація (параметр порогу можна змінювати вручну)
_, binary_mask = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

# Знаходження контурів об'єкту на масці
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Обрізання об'єкту з оригінального зображення за допомогою маски
object_mask = np.zeros_like(original_image)
cv2.drawContours(object_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
object_image = cv2.bitwise_and(original_image, object_mask)

# Збереження результатів
cv2.imwrite("WB.jpg", gray_image)
cv2.imwrite("Mask.jpg", binary_mask)
cv2.imwrite("ob_on_WB.jpg", object_image)

# Повторіть ті самі кроки для кольорового зображення, якщо потрібно
# Конвертування в ч/б варіант
color_gray_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

# Глобальна бінаризація (параметр порогу можна змінювати вручну)
_, color_binary_mask = cv2.threshold(color_gray_image, 150, 255, cv2.THRESH_BINARY)

# Збереження результатів
cv2.imwrite("color_WB_pic.jpg", color_gray_image)
cv2.imwrite("color_mask.jpg", color_binary_mask)