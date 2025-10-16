🪷 Mudra Recognition of Sattriya Folk Dance 

🌟 Overview:
This project recognizes and classifies *Sattriya Folk Dance mudras* in real time using *Computer Vision* and *Machine Learning*.  
It leverages *MediaPipe* for 3D hand landmark detection and *OpenCV* for live video processing.

🎯 Objective:
To merge Indian Classical Dance Heritage with modern AI — demonstrating how traditional hand gestures (mudras) can be digitally recognized, preserved, and analyzed.

🧠 Tech Stack:
- Python
- OpenCV
- MediaPipe
- NumPy


⚙️ Architecture Workflow:

🎥 Webcam Input  
       ↓  
🖐️ MediaPipe Hands (21 Landmark Extraction)  
       ↓  
🧮 Feature Vector (x, y coordinates normalized)  
       ↓  
📊 Output: Mudra Name + Confidence Score  
       ↓  
🖼️ Display: Real-Time Frame Overlay (OpenCV)


⚙️ System Workflow:
1. Data Collection: Captured images of different Sattriya mudras using webcam.
2. Landmark Extraction: MediaPipe Hands model extracts 21 keypoints per hand.
3. Feature Engineering: Normalized landmark coordinates using NumPy.
4. Real-Time Recognition: Integrated trained model into OpenCV live feed.
5. Performance Evaluation: Achieved ~90% accuracy across 12 unique mudras.

💡 Future Scope: 
- Extend recognition to other Indian classical dance forms  
- Build interactive learning tool for students and researchers  
- Explore use in sign language interpretation

📚 Learnings:
- Real-time computer vision pipeline integration  
- Landmark-based feature engineering  

>> “Technology should not replace tradition — it should help preserve it.”
