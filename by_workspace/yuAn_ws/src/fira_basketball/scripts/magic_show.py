#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gtts import gTTS
import os
import cv2 
import numpy as np
import time

def magic_skill():
    
    output_dir = "cropped_contours"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("無法打開相機")
        exit()

    cut_bount = (np.array([0, 0, 0]), np.array([100, 255, 50])) #green

    def cut_and_scale_pic(img,color_bount,scale_percent=100):
        # 將畫面轉換為 HSV 色彩空間
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        
        lower_black = color_bount[0]      
        upper_black = color_bount[1] 

        # print(lower_black,upper_black)


        mask = cv2.inRange(hsv, lower_black, upper_black)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3)))

        cv2.imshow('Black Mask Cut', mask)

        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        cropped = None
        have_cropped = False
        for i, contour in enumerate(contours):
           
            x, y, w, h = cv2.boundingRect(contour)
            
            if 25000 > cv2.contourArea(contour) > 5000:  # 設定面積閾值
                have_cropped = True
                
                padding = 10  
                x_start = max(x - padding, 0)
                y_start = max(y - padding, 0)
                x_end = min(x + w + padding, img.shape[1])
                y_end = min(y + h + padding, img.shape[0])

                
                cropped = img.copy()
                cropped = cropped[y_start:y_end, x_start:x_end]

                width = int(cropped.shape[1] * scale_percent / 100)
                height = int(cropped.shape[0] * scale_percent / 100)
                resized_cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)

                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(img, str(cv2.contourArea(contour)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break
        if not have_cropped:
            return False, img

        return True, resized_cropped

    def find_ans(img,bount,det_yellow = False,dec_middle = True,approxPolyDP_val = 10):
        ans = "unknown"

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (600, 300))
            cv2.imshow('gray', gray)
            # print("gray size",gray.shape)

            center_x = gray.shape[1] // 2 #300


            
            _, mask = cv2.threshold(gray, bount, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3)))
            cv2.imshow('Black Mask Ans', mask)

            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                contour = max(contours, key=cv2.contourArea)
                contour = cv2.approxPolyDP(contour, approxPolyDP_val ,True)
                # cv2.drawContours(gray, contour, -1, (255), 5)
                
                sorted_points = sorted(contour[:, 0, :], key=lambda point: point[1], reverse=True)
                bottom_points = sorted_points[:5]
                # print(bottom_points)

                top_point = sorted_points[4]
                sec_top_point = sorted_points[3]
                cv2.circle(gray, top_point , 10, (255), 1)
                # cv2.putText(gray, str(abs(top_point[1])), top_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
                cv2.circle(gray, sec_top_point , 10, (255), 1)
                # cv2.putText(gray, str(abs(sec_top_point[1])), sec_top_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

                print("top",abs(top_point[1] - sec_top_point[1]))
                # print("top_point",top_point)
                # print("sec_top_point",sec_top_point)

                if (abs(top_point[1] - sec_top_point[1]) > 20):
                    # if det_yellow:
                    #     ans = "yellow"
                    print("cut")
                    bottom_points = bottom_points[:4]


                for point in bottom_points:
                    cv2.circle(gray, tuple(point), 5, (255), -1)
                    # cv2.putText(gray, str(abs(point[0] - center_x)), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

                sorted_points = sorted(bottom_points, key=lambda point: abs(point[0] - center_x))
                middle_point = sorted_points[0]
                cv2.circle(gray, middle_point , 15, (255), 1)
                other_points = sorted_points[1:]

                left = 0
                right = 0
                dir = 1
                if (middle_point[0] > center_x) and dec_middle:
                    dir = -1
                for point in other_points:
                    cv2.circle(gray, tuple(point), 5, (255), -1)
                    if point[0] < center_x:
                        left += dir
                    else:
                        right += dir
                if left > right:
                    ans = "red"
                elif right == left:
                    if det_yellow:
                        ans = "yellow"
                else:
                    ans = "blue"
            cv2.imshow('img with contour', gray)
            return ans
        except Exception as e:
            print(e)
            return ans






    while True:
        
        ret, img = cap.read()
        if not ret:
            print("無法讀取相機畫面")
            break

        ret,cut_pic = cut_and_scale_pic(img,cut_bount)

        if ret:
            ans = find_ans(cut_pic,bount = 90,det_yellow = True)
            cv2.putText(img, ans, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print("ans",ans)


        
        cv2.imshow('Camera Feed', img)
        # cv2.imshow('Black Mask', mask)
        if ret:
            cv2.imshow('cut_pic', cut_pic)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()

    return ans


def text_to_speech(text, lang, filename):
    
    try:
       
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        print(f"語音已儲存為 {filename}")

        
        
    except Exception as e:
        print(f"發生錯誤: {e}")

def play_audio(filename):
    
    if os.name == 'nt':  
        os.system(f'start {filename}')
    elif os.name == 'posix':  
        os.system(f'afplay {filename}' if 'darwin' in os.uname().sysname else f'play {filename}')


if __name__ == "__main__":
    input_text = "Before I perform, I need to explain what I'm going to do. First I need to ask someone to do me a favor and help me choose a randomly colored figure to put in it. Then he put away the remaining colored figures, leaving only the black coffin on the table.。"
    
    text_to_speech(input_text, lang='zh', filename='first_line.mp3')
    play_audio("first_line.mp3")

    time.sleep(35)

    input_text2 = "If you have put the figure into the black coffin, please say start"
    text_to_speech(input_text2, lang='zh', filename='second_line.mp3')
    play_audio("second_line.mp3")

    time.sleep(10)

    ans = magic_skill()

    if ans == "yellow":
        input_text3 = "我認為你放入的是黃色的人偶"
        text_to_speech(input_text3, lang='zh', filename='third_line.mp3')
    elif ans == "blue":
        input_text3 = "我認為你放入的是藍色的人偶"
        text_to_speech(input_text3, lang='zh', filename='third_line.mp3')
    elif ans == "red":
        input_text3 = "我認為你放入的是紅色的人偶"
        text_to_speech(input_text3, lang='zh', filename='third_line.mp3')
    
    play_audio("third_line.mp3")
