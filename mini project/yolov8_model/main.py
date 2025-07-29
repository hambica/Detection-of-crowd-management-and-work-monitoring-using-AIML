from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

global filename, model
labels = ['Person', 'Crowd']
CONFIDENCE_THRESHOLD = 0.3
ALERT_THRESHOLD = 30
GREEN = (0, 255, 0)
global count

main = tkinter.Tk()
main.title("Detection of Crowd Management & Work Monitoring using AI & ML")
main.geometry("1300x1200")

# Function to load YOLOv8 model
def loadModel():
    global filename, model
    text.delete('1.0', END)
    model = YOLO("../best.pt")  # corrected path
    text.insert(END, "YOLOv8 Model Loaded\n")

# Function for detecting crowd in images
def imageDetection():
    global model, count
    count = 0
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="../")  
    text.insert(END, filename + " loaded\n\n")
    frame = cv2.imread(filename)
    frame = cv2.resize(frame, (700, 600))
    detections = model(frame)[0]

    for data in detections.boxes.data.tolist():
        confidence = data[4]
        cls_id = data[5]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        if cls_id == 0 or cls_id == 1:
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(frame, str(count), (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            count += 1

    if count > ALERT_THRESHOLD:
        messagebox.showwarning("Crowd Alert", f"⚠️ High crowd detected: {count} people!")

    cv2.imshow("Object Detection from Image", frame)
    cv2.waitKey(0)

# Function for detecting crowd in videos
def videoDetection():
    global model, count
    count = 0
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="../")  # corrected path
    text.insert(END, filename + " loaded\n\n")
    video_cap = cv2.VideoCapture(filename)

    while True:
        count = 0
        ret, frame = video_cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (700, 600))
        detections = model(frame)[0]

        for data in detections.boxes.data.tolist():
            confidence = data[4]
            cls_id = data[5]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
            if cls_id == 0 or cls_id == 1:
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                cv2.putText(frame, str(count), (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                count += 1

        if count > ALERT_THRESHOLD:
            messagebox.showwarning("Crowd Alert", f"⚠️ High crowd detected: {count} people!")

        cv2.imshow("Object Detection from Video", frame)
        if cv2.waitKey(50) == ord("q"):
            break

    video_cap.release()
    cv2.destroyAllWindows()

# (Optional) Disable or skip training graph for now
def graph():
    # You can later update the path to a real graph if needed
    # graph_img = cv2.imread('../results.png')
    # graph_img = cv2.resize(graph_img, (800, 600))
    # cv2.imshow("YoloV8 Training Graph", graph_img)
    # cv2.waitKey(0)
    messagebox.showinfo("Graph", "Training graph not available.")

def close():
    main.destroy()

# GUI Layout
font = ('times', 16, 'bold')
title = Label(main, text='Crowd Management, Crime Prevention & Work Monitoring using AI & ML')
title.config(bg='greenyellow', fg='dodger blue')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
loadButton = Button(main, text="Generate & Load YOLOv8 Model", command=loadModel)
loadButton.place(x=50, y=550)
loadButton.config(font=font1)

imageButton = Button(main, text="Crowd Management from Images", command=imageDetection)
imageButton.place(x=330, y=550)
imageButton.config(font=font1)

videoButton = Button(main, text="Crowd Management from Videos", command=videoDetection)
videoButton.place(x=630, y=550)
videoButton.config(font=font1)

graphButton = Button(main, text="YOLOv8 Training Graph", command=graph)
graphButton.place(x=50, y=600)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=330, y=600)
exitButton.config(font=font1)

main.config(bg='LightSkyBlue')
main.mainloop()
