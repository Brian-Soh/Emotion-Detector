//
//  main.cpp
//  Emotion-Detection
//
//  Created by Brian Soh on 2025-09-20.
//
#include "main.h"

using namespace cv;
using namespace std;

int main() {
    
    CascadeClassifier faceCascade;
    faceCascade.load("/Users/briansoh/Downloads/Resources/haarcascade_frontalface_default.xml");
    
    if (faceCascade.empty()) {
        cerr << "XML file not loaded" << endl;
        return -1;
    }
    
    dnn::Net net = dnn::readNetFromONNX("Models/emotion_detector.onnx");

    
    Mat img;
    VideoCapture cap(0);
    
    vector<string> emotionLabels = {
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral"
    };

    
    while (true) {
        cap.read(img);
        
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        
        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 10);
        for (int i = 0; i < faces.size(); i++) {
            Mat compressed;
            Mat faceROI = img(faces[i]);
            resize(faceROI, compressed, Size(224, 224));
            compressed.convertTo(compressed, CV_32F, 1.0/255.0);  // normalize to [0,1]
            
            Mat blob = dnn::blobFromImage(compressed, 1.0, Size(224, 224), Scalar(), false, false);

            net.setInput(blob);
            Mat output = net.forward();

            // Get predicted class
            Point classIdPoint;
            double confidence;
            minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);
            int predicted = classIdPoint.x;

            cout << "Predicted emotion: " << emotionLabels[predicted]
                 << " with confidence " << confidence << endl;
            
            rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
            
            string label = "Face " + to_string(i + 1);
            Point textOrg(faces[i].x, faces[i].y - 10);

            // Draw the text
            putText(img, label, textOrg, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 255), 2);
        }
        
        imshow("Webcam", img);
        
        if (waitKey(1) >= 0) break;
    }
    
    return 0;
}
