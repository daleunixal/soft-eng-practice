# МОДЕЛЬ "КЛАССИФИКАЦИЯ ЭКГ" #

## О МОДЕЛИ ## 

https://huggingface.co/gianlab/swin-tiny-patch4-window7-224-finetuned-ecg-classification
Модель работает с изображением ЭКГ размером 224px, 224px и имеет возможность классифицировать данные по следующим типам: 

Классифицированные данные:
* N: Нормальный ритм
* S: Суправентрикулярная экстрасистолия.
* V: Преждевременное сокращение желудочков.
* F: Слияние желудочкового и нормального сокращений.
* Вопрос: Неклассифицируемый бит
* М: инфаркт миокарда

## Работа с скриптом ##

Необходимо в деректорию со скриптом положить файл с названием 'img.png' фотографии со сотношением сторон 1:1 и запустить скрипт

