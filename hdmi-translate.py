#!/bin/env python3

#  This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv2
import pytesseract
import numpy as np
from translate import Translator
import tensorflow as tf
import textwrap
import pyaudio
import threading
import difflib
import re
import os
import json


# Translate Tesseract
# en eng
# ja jpn
try:
    params = json.load(open(f'.{os.sep}params.json', 'r'))
except:
    params = {
        'language_from_tesseract': 'eng',
        'language_from_translate': 'en',
        'language_to': 'ru',
        'use_srgan': False,

        'video_device_index': 0,
        'output_width': 1920,
        'output_height': 1080,
        'border': 50,

        'audio_device_pattern': 'USB Audio',
        'channels': 2
    }
    json.dump(params, open(f'.{os.sep}params.json', 'w'), indent=4)

debug = False
debug_image = f'.{os.sep}jrpg.jpg'

# Для использования нейросети SRGAN для апскейла текста перед распознаванием нужно скачать
# модель по ссылке https://tfhub.dev/captain-pool/esrgan-tf2/ и распаковать её в папку
# esrgan рядом со скриптом.
use_srgan = bool(params['use_srgan'])
if use_srgan:
    try:
        srgan = tf.keras.models.load_model(f'.{os.sep}esrgan')
    except:
        use_srgan = False

language_from_tesseract = params['language_from_tesseract']
language_from_translate = params['language_from_translate']
language_to = params['language_to']

stop_flag = False
input_device_index = 0
sample_rate = 44100
audio_device_name = params['audio_device_pattern']
p = pyaudio.PyAudio()

device_count = p.get_device_count()
sample_rate = int(p.get_device_info_by_index(0)['defaultSampleRate'])
for i in range(device_count):
    device_info = p.get_device_info_by_index(i)
    print(f"Device {i}: {device_info['name']}")
    if audio_device_name in device_info['name']:
        input_device_index = i
        sample_rate = int(p.get_device_info_by_index(i)['defaultSampleRate'])

# Настройки захвата видео
VIDEO_DEVICE = params['video_device_index']    # **1** - Внешняя камера или USB-камера.
VIDEO_WIDTH = params['output_width']
VIDEO_HEIGHT = params['output_height']
BORDER = params['border']

# Настройки захвата аудио.
CHUNK_SIZE = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = params['channels']
stream = p.open(format=FORMAT, 
                channels=CHANNELS, 
                rate=sample_rate, 
                input=True, 
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK_SIZE)
output = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=sample_rate,
                output=True)   


def capture_audio(callback):
    while not stop_flag:
        data = stream.read(CHUNK_SIZE)
        if not data:
           break
        callback(data)
    stream.stop_stream()
    stream.close()
    p.terminate()


def play_audio(data):
    output.write(data)


# Начинаем захват и воспроизведение аудио.
if not debug:
    capture_thread = threading.Thread(target=capture_audio, args=(play_audio,))
    capture_thread.start()


mouse_action = 0
mouse_x = 0
mouse_y = 0
top_left = (0, 0)
bottom_right = (VIDEO_HEIGHT, VIDEO_WIDTH)


def mouse_callback(event, x, y, flags, param):
    global mouse_action, top_left, bottom_right, mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        if mouse_action == 0:
            top_left = (y, x)
            bottom_right = top_left
            mouse_x, mouse_y = x, y
            mouse_action = 1
        elif mouse_action == 1:
            bottom_right = (y, x)
            tl = (min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1]))
            br = (max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1]))
            top_left = tl
            bottom_right = br
            mouse_action = 0
    elif mouse_action == 1 and event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


translator = Translator(language_to, language_from_translate)
window_name = 'HDMI translator'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.resizeWindow(window_name, VIDEO_WIDTH, VIDEO_HEIGHT)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window_name, mouse_callback)


# Функция для автоматического объединения текстов
def merge_texts(text1, text2):
    merged_text = ''
    matcher = difflib.SequenceMatcher(None, text1, text2)
    differences = matcher.get_opcodes()
    for opcode, _, _, j1, j2 in differences:
        if opcode == 'equal' or opcode == 'insert':
            # Общая часть или добавленная часть во втором тексте
            merged_text += text2[j1:j2]
        elif opcode == 'delete':
            # Удаленная часть из первого текста
            pass  # Пропускаем удаленную часть
        elif opcode == 'replace':
            # Замененная часть текста
            merged_text += text2[j1:j2]

    return merged_text


def remove_special_characters(text):
    if language_from_tesseract == 'eng':
        pattern = r'[^a-zA-Z0-9.,!?\s]'
        text = re.sub(pattern, '', text)
    elif language_from_tesseract == 'jpn':
        text = text.replace(' ', '')
        pass
    return text


def translate_image(frame):
    if mouse_action == 0:
        try:
            frame = frame[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        except:
            pass
    if use_srgan:
        # Загрузка изображения
        srgan_frame = tf.cast(frame, tf.float32) / 255.0
        srgan_frame = tf.expand_dims(srgan_frame, axis=0)
        srgan_frame = srgan.predict(srgan_frame)
        srgan_frame = tf.squeeze(srgan_frame, axis=0)
        srgan_frame = tf.clip_by_value(srgan_frame, 0, 1) * 255.0
        frame = tf.cast(srgan_frame, tf.uint8).numpy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    inverted = 255 - gray
    text = pytesseract.image_to_string(gray, lang=language_from_tesseract)
    text_inverted = pytesseract.image_to_string(inverted, lang=language_from_tesseract)
    text = merge_texts(text, text_inverted)
    text = remove_special_characters(text)    
    return translator.translate(text)


def upscale_image(image):
    height, width = image.shape[:2]
    scale = max(1, min(VIDEO_HEIGHT // height, VIDEO_WIDTH // width))
    new_width = int(width * scale)
    new_height = int(height * scale)    
    new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    # Создание черного фона
    bg = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    left_border = (VIDEO_WIDTH - new_width) // 2
    top_border = (VIDEO_HEIGHT - new_height) // 2
    # Размещение уменьшенного изображения на фоне
    bg[top_border:top_border+new_height, left_border:left_border+new_width] = new_image
    return bg


def overlay_text(image, text, font_face=cv2.FONT_HERSHEY_COMPLEX, font_scale=1.0, font_thickness=2):
    image_height = image.shape[0]
    
    # Оверлей.
    x, y = 100, int(3*image_height/5)
    alpha = 0.8
    overlay = image.copy()
    cv2.rectangle(overlay, (BORDER, y), (VIDEO_WIDTH - BORDER, VIDEO_HEIGHT - BORDER), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Текст.
    y += 50
    i = 0
    wrapped_text = textwrap.wrap(text, width=80)
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font_face, font_scale, font_thickness)[0]
        gap = textsize[1] + 10
        y += i * gap
        i += 1
        cv2.putText(image, line, (x, y), font_face, font_scale, (255,255,255), font_thickness, lineType = cv2.LINE_AA)

    return image

# # Создание объекта VideoCapture для захвата потока видео с устройства захвата HDMI
if not debug:
    capture = cv2.VideoCapture(VIDEO_DEVICE, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    # # Проверка успешного открытия потока видео
    if not capture.isOpened():
        print("Не удалось открыть устройство захвата.")
        exit()

translate_mode = False
translated_text = None

# Цикл для чтения и отображения кадров из видеопотока
while True:
    # Захват кадра из видеопотока
    if debug:
        ret, frame = True, cv2.imread(debug_image)
    else:
        ret, frame = capture.read()

    # Если кадр успешно получен
    if ret:
        if frame.shape[:2] != (VIDEO_HEIGHT, VIDEO_WIDTH):
            frame = upscale_image(frame)
        if translate_mode:
            if translated_text is None:
                translated_text = translate_image(frame)
            frame = overlay_text(frame, translated_text)
        if mouse_action == 1:
            cv2.rectangle(frame, (top_left[1], top_left[0]), (mouse_x, mouse_y), (255, 0, 0), -1)
        cv2.imshow(window_name, frame)

    # Обработка ввода
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        translate_mode = not translate_mode
        translated_text = None

# Освобождение ресурсов
if not debug:
    capture.release()
    stop_flag = True
    capture_thread.join()
cv2.destroyAllWindows()
