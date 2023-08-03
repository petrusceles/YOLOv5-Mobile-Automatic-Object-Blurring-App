# SINGLE OBJECT AUTOMATIC BLURRING USING INT8 QUANTIZED TFLITE OBJECT DETECTION MODEL
**Here is my final undergraduate project to create a computer vision real-time application**

- Using image analysis use case from CameraX API to get the frame data
- Parallel processing so that it can run with more than 20 fps for **yolov5n6 512x512 int8 tflite model** on POCO F3 [Snapdragon 870] but with some delay
- Utilizing SORT Object Tracking to increase positive detection
- You can choose between grayscale or histogram equalization on preprocessing

## IMPORTANT NOTES!!

- I've tried **yolov5n6 512x512 int8 tflite model** on Redmi Note 10s. If I change the input size to **448x448**, it works. I do not know why.
- This app will save the recording or frame and object detection data acquisition result in the application root folder, located in /Android
- The save folder name is based on the model name. You can add prefixes from the configuration page
- While recording, you should not rotate the smartphone if the orientation is not locked
- I will fix the bugs if I have time, thank you

## RELATED FILES
- License plate detection model (Yolov5n6 int8 512x512): https://drive.google.com/file/d/10tVmOHFGPF0fxkO29KDdzYErI0OLitTn/view?usp=sharing
- Video result example: https://drive.google.com/file/d/1djGY97TuidRbModpCLGPP8ZZxxs9hdHB/view?usp=sharing

## UI
- Configuration Page

![Copy of DesainSistemKonfigurasi](https://github.com/petrusceles/Automatic-Object-Blurring-App/assets/90450258/52893793-9289-4e8d-8f3f-a5fe40895979)


- Camera Page

![CameraUI](https://github.com/petrusceles/Automatic-Object-Blurring-App/assets/90450258/9399b5a7-1ef2-41d6-811f-310fa50672c6)

