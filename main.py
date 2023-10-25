from transformers import pipeline
from keras.preprocessing import image

cls = pipeline("image-classification", model="gianlab/swin-tiny-patch4-window7-224-finetuned-ecg-classification")

img = image.image_utils.load_img('./img.png', target_size=(224, 224))

result = cls(img)

print(max(result, key=lambda x: x['score']))
