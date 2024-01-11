import base64
import io
import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from imutils.contours import sort_contours
from keras.models import load_model
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from skimage.metrics import structural_similarity
from skimage.transform import resize


@api_view(['GET', 'POST'])
def check(request):
    response_data = {
        'status': True,
        'message': 'Connected Successfully',
        'data': None
    }

    if request.method == 'GET':
        return Response(response_data, status=status.HTTP_200_OK)

    elif request.method == 'POST':
        return Response(response_data, status=status.HTTP_201_CREATED)


@api_view(['GET', 'POST'])
def predict(request):
    if request.method == 'GET':
        response_data = {
            'status': True,
            'message': 'Api connected succesfully without any data',
            'data': None
        }
        return Response(response_data, status=status.HTTP_200_OK)

    elif request.method == 'POST':
        try:
            img_encoded = request.data['img']
            img = Image.open(io.BytesIO(base64.b64decode(img_encoded)))
            img.save('outputs/1_sketch_from_app.png')

            img = cv2.imread('outputs/1_sketch_from_app.png')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("outputs/2_gray_noise_remove.png", gray)

            (_, blackAndWhiteImage) = cv2.threshold(
                gray, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite("outputs/3_black_and_white_noise_remove.png",
                        blackAndWhiteImage)

            blur = cv2.GaussianBlur(
                blackAndWhiteImage, (0, 0), sigmaX=33, sigmaY=33)
            cv2.imwrite("outputs/4_blur_noise_remove.png", blur)

            divide = cv2.divide(blackAndWhiteImage, blur, scale=255)
            cv2.imwrite("outputs/5_divide_noise_remove.png", divide)

            thresh = cv2.threshold(
                divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            cv2.imwrite("outputs/6_thresh_noise_remove.png", thresh)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite("outputs/7_morph_noise_remove.png", morph)

            img_gray = gray
            edged = cv2.Canny(img_gray, 30, 150)
            cv2.imwrite("outputs/8_all_canny_detect.png", edged)

            contours = cv2.findContours(
                edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sort_contours(contours, method="left-to-right")[0]

            model = load_model('./static/sketch_recognition_cnn.keras')

            labels = ['apple', 'banana', 'candle', 'donut', 'envelope',
                      'flower', 'ice_cream', 'leaf', 'mug', 'umbrella']
            area_max = -9999
            for i, c in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(c)
                if x > 0 and y > 0 and w > 20:
                    roi = img[y:y+h, x:x+w]
                    roi = img_gray[y:y+h, x:x+w]
                    thresh = cv2.threshold(
                        roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                    (th, tw) = thresh.shape
                    if tw > th:
                        thresh = imutils.resize(thresh, width=32)
                    if th > tw:
                        thresh = imutils.resize(thresh, height=32)
                    (th, tw) = thresh.shape

                    dx = int(max(0, 32 - tw)/2.0)
                    dy = int(max(0, 32 - th) / 2.0)
                    padded = cv2.copyMakeBorder(
                        thresh, top=dy, bottom=dy, left=dx, right=dx, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    padded = cv2.resize(padded, (32, 32))
                    padded = np.array(padded)
                    padded = padded/255.
                    padded = np.expand_dims(padded, axis=0)
                    padded = np.expand_dims(padded, axis=-1)
                    pred = model.predict(padded)
                    pred = np.argmax(pred, axis=1)
                    label = labels[pred[0]]
                    print('>>>>The {} no word is : {}'.format(i, label))

                    area = w * h
                    if (area_max < area):
                        area_max = area
                        prediction = label
                        cropped_image = img[y:y + h, x:x + w]
                        cv2.imwrite(
                            "outputs/9_cropped_image_of_prediction.png", cropped_image)

                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(img, label, (x-5, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            plt.figure(figsize=(10, 10))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite("outputs/10_system_prediction.png", img)
            plt.imshow(img)
            plt.savefig('outputs/11_with_axis.png')
            plt.axis('off')
            plt.savefig('outputs/12_without_axis.png')

            apple = './contents/apple_ref.jpg'
            banana = './contents/banana_ref.jpg'
            candle = './contents/candle_ref.jpg'
            donut = './contents/donut_ref.jpg'
            envelope = './contents/envelope_ref.png'
            flower = './contents/flower_ref.png'
            ice_cream = './contents/ice_cream_ref.jpg'
            leaf = './contents/leaf_ref.jpg'
            mug = './contents/mug_ref.png'
            umbrella = './contents/umbrella_ref.jpg'

            prediction_to_path = {
                "apple": apple,
                "banana": banana,
                "candle": candle,
                "donut": donut,
                "envelope": envelope,
                "flower": flower,
                "ice_cream": ice_cream,
                "leaf": leaf,
                "mug": mug,
                "umbrella": umbrella
            }

            if prediction in prediction_to_path:
                img1_path = prediction_to_path[prediction]
            else:
                img1_path = None

            img2_path = './outputs/9_cropped_image_of_prediction.png'

            if not os.path.isfile(img1_path) or not os.path.isfile(img2_path):
                response_data = {
                    "success": False,
                    "message": "Image files not found"
                }
                return Response(response_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                img1 = cv2.imread(img1_path, 0)
                img2 = cv2.imread(img2_path, 0)

                if img1 is None or img2 is None:
                    response_data = {
                        "success": False,
                        "message": "Image not found"
                    }
                    return Response(response_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                else:
                    orb_similarity = orb_sim(img1, img2)
                    orb_similarity = round(orb_similarity * 100, 2)

                    img2_resized = resize(
                        img2, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)
                    ssim = structural_sim(img1, img2_resized)
                    ssim = round(ssim * 100, 2)

            response_data = {
                "success": True,
                "message": "Successfully saved the sketch",
                "prediction": prediction,
                "orb": orb_similarity,
                "ssim": ssim
            }
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            response_data = {
                "success": False,
                "message": str(e),
            }
            return Response(response_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def orb_sim(img1, img2):
    orb = cv2.ORB_create()
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desc_a, desc_b)
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


def structural_sim(img1, img2):
    sim, _ = structural_similarity(img1, img2, full=True, data_range=1.0)
    return sim
