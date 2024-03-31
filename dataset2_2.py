import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained ResNet50 model โหลดโมเดล RestNet50
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

#แตกข้อมูลของรูปภาพออกมาเป็นอาร์เรย์ 1 มิติ
# Function to extract features from an image
def extract_features(img_path, model):
    #โหลดรูปภาพเก็บไว้ในตัวแปร img
    img = image.load_img(img_path, target_size=(224, 224))
    #แตกข้อมูลภาพเก็บลงใน Vector
    img_data = image.img_to_array(img)
    #เตรียมข้อมูลให้พร้อมสำหรับการประมวลผลโดยการเพิ่ม axis 
    img_data = np.expand_dims(img_data, axis=0)
    #เรียกฟังก์ชัน preprocess เพื่อเตรียมข้อมูลให้ใช้กับ model ResNet50 ได้
    img_data = preprocess_input(img_data)
    #ทำนายและเก็บลักษณะเฉพาะของรูปภาพและเก็บไว้ในตัวแปร Features
    features = model.predict(img_data)
    #แปลงข้อมูลเป็น Array 1 มิติและส่งค่ากลับ
    return features.flatten()

# Function to calculate similarity between two images based on cosine similarity 
#คำนวณความเหมือนระหว่าง 2 รูปภาพ
def calculate_similarity(img_features1, img_features2):
    return cosine_similarity([img_features1], [img_features2])[0][0]

# Path to the directory containing pet images
data_dir = "dataset"

# Example user input: path to the desired pet image
user_input_path = "slid2.jpg"

# Extract features for the user input image
#เรียกใช้ฟังก์ชัน แตกข้อมูลของรูปภาพที่ user ส่งเข้ามา ออกมาเป็นอาร์เรย์ 1 มิติ
user_input_features = extract_features(user_input_path, model)

# Load pet images and their paths into a DataFrame
pet_data_list = []

#วนลูปการเรียกไฟล์ในโฟลเดอร์ Dataset ให้ครบทุกโฟลเดอร์ แต่ละโฟลเดอร์คือที่เก็บรูปภาพของสัตว์แต่ละตัว แต่ละตัวมีได้หลายรูป
for pet_name in os.listdir(data_dir):
    # ตรงนี้ไม่ต้องสนใจ
    # ignore โฟลเดอร์ที่ชื่อว่า .DS_Store files. ไม่ต้องดำเนินการอะไรกับโฟลเดอร์นี้ (ใช้แก้ปัญหากรณีรันบนเครื่อง Mac เท่านั้น)
    # Ignore .DS_Store files
    if pet_name == ".DS_Store":
        continue
    pet_name_dir = os.path.join(data_dir, pet_name)
    max_similarity = -1  # Initialize max similarity for each pet name
    max_img_path = ""    # Initialize image path with max similarity
    #เมื่อเจอโฟลเดอร์แรก จะทำการวนเรียกไฟล์ในโฟลเดอร์อีกครั้ง โดยการเรียกไฟล์คือการเรียกฟังก์ชัน Extract นั่นก็คือแตกข้อมูลของรูปภาพออกมาเป็นอาร์เรย์ 1 มิติ และทำการเปรียบเทียบรูปภาพดังกล่าว
    #กับรูปภาพที่ผู้ใช้ได้ทำการส่งเข้ามา อย่างที่บอกว่า สัตว์แต่ละตัว จะมีได้หลายรูป หลายมุม อาจจะมีรูปที่เหมือนรูปที่ผู้ใช้ใส่มาบ้าง หรือไม่เหมือนบ้างก็มี่ 
    #เราจึงทำการเลือกออกมา 1 รูป ที่เหมือนที่สุด และเก็บสถิติไว้ใน list
    for img_file in os.listdir(pet_name_dir):
        img_path = os.path.join(pet_name_dir, img_file)
        features = extract_features(img_path, model)
        #คำนวณความเหมือนของรูปที่กำลังดำเนินการอยู่ กับรูปที่ผู้ใช้ได้ทำการส่งเข้ามา
        similarity = calculate_similarity(features, user_input_features)
        #ถ้ามีรูปที่ได้ค่าความเหมือนมากกว่า ก็ให้เก็บค่าของรูปนั้นไปใช้ 
        if similarity > max_similarity:
            max_similarity = similarity
            max_img_path = img_path
    #insert ชื่อสัตว์ และค่าความเหมือนเข้าในลิสต์
    pet_data_list.append(
        {"image_path": max_img_path, "pet_name": pet_name, "similarity": max_similarity}
    )



# Create DataFrame from the list of dictionaries
# เมื่อวนทำจนครบทุกโฟลเดอร์(วนเปรียบเทียบกับสัตว์ครบทุกตัว) แล้ว จะทำการแปลง List เป็น Dataframe
pet_data = pd.DataFrame(pet_data_list)

# Sort by similarity score and select the top 5 recommendations
#กำหนดตัวแปรที่เก็บข้อมูล 5 อันดับของสัตว์ที่ได้คะแนนความเหมือนสูงสุด
top_5_recommended_pets = pet_data.nlargest(5, 'similarity')

#print ค่าที่ Terminal
print("Top 5 recommended pets:")
print(top_5_recommended_pets[["image_path", "pet_name", "similarity"]])
