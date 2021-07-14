#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("+==============================================================+")
print("|>                                                            <|")
print("|>      Application Name     :    Face predication            <|")
print("|>      Developer            :    Satya Mishra                <|")
print("|>      Investigator         :    Mr. Pankaj Shukla           <|")
print("|>      Email                :    satyamishra559@gmail.com    <|")
print("|>      Language Used        :    Python 3 and Python 2       <|")
print("|>      Based On             :    Machine Learning            <|")
print("|>                                                            <|")
print("+==============================================================+")
print("[*]importing required module Please wait...")
import pandas as pd
import numpy as np

# Algorithem import
from skimage.transform import rescale
from skimage.io import imshow,imread
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

lr=LogisticRegression()
svm=SVC()
gnb=GaussianNB()
tree=DecisionTreeClassifier()
knn=KNeighborsClassifier()
print("[+] All required module imported successfully !")


# In[2]:


# csv file name to reading
csv_file=input("Enter CSV file which containe image information (Path like 'D:\\image-classification\\train.csv') >> ")


# In[3]:


# Reading csv
img_data=pd.read_csv(csv_file)


# In[4]:


folder=input("Enter path where store training images (Path like 'D:\\image-classification\\') '>> ")


# In[5]:


# Data Preprocessing doing here...
print("\n[*] Your data going to preprocessing it's take some time ...\n")
row_20=[]
size=float(input("How make size your want to reduce image we recomanded(0.25) between (0.0-1.0) >> "))
for i in range(img_data.shape[0]):
    print("[*]",i," image Loading...")
    img_name=img_data.iloc[i].image
    path=folder+img_name
    img=imread(path,as_gray=True)
    print("[*] reducing the size of image ...")
    r_img=rescale(img,size)
    print("[*] Flatting the image into 1 row ...")
    f_img=np.reshape(r_img,r_img.shape[0]*r_img.shape[1])
    print("[*]",img_name," are saving...")
    row_20.append(f_img)
    print("[+]",img_name,"Process complete ... \n\n")
print("[*] Making DataFrame, it's take some time please wait...")
X_train=pd.DataFrame(row_20)
print("[+] DataFrame maked successfully !")


# In[6]:


print("\n[*] Making Outcome Training data ...")
y_train=img_data.label
print("[+] Done ! \n")


# In[7]:


test_csv=input("Enter CSV file path which contain testing data information (Path like 'D:\\image-classification\\test.csv') >> ")
print("[*] Reading CSV file...")
test_csv=pd.read_csv(test_csv)
print("[+] Done !")


# In[8]:


test_folder=input("Enter your Testing image location (Path like 'D:\\image-classification\\') >> ")


# In[9]:


test_row=[]
for i in range(test_csv.shape[0]):
    print("[*]",i+1," image Loading...")
    img_name=test_csv.iloc[i].image
    path=folder+img_name
    img=imread(path,as_gray=True)
    print("[*] reducing the size of image ...")
    r_img=rescale(img,size)
    print("[*] Flatting the image into 1 row ...")
    f_img=np.reshape(r_img,r_img.shape[0]*r_img.shape[1])
    print("[*]",img_name," are saving...")
    test_row.append(f_img)
    print("[+]",img_name,"Process complete ... \n\n")
print("[*] Making DataFrame, it's take some time please wait...")
X_test=pd.DataFrame(test_row)
print("[+] DataFrame maked successfully !")


# In[10]:


print("\n[*] Creating Outcome testing data ...")
y_test=test_csv.label
print("[+] Done ! \n")


# In[11]:


# X_train.shape


# In[12]:


# X_test.shape


# In[13]:


# y_train.shape


# In[14]:


# y_test.shape


# In[15]:


print("[-] Now we store all data...")


# In[16]:


print("[*] fiting Training data into LogisticRegression Algorithem..")
lr.fit(X_train,y_train)
print("[+] Done !")
print("[*] fiting Training data into SVM Algorithem..")
svm.fit(X_train,y_train)
print("[+] Done !")
print("[*] fiting Training data into GaussianNB Algorithem..")
gnb.fit(X_train,y_train)
print("[+] Done !")
print("[*] fiting Training data into DecisionTreeClassifier Algorithem..")
tree.fit(X_train,y_train)
print("[+] Done !")
print("[*] fiting Training data into KneighborsClassifier Algorithem..")
knn.fit(X_train,y_train)
print("[+] Data fitted into all algrithem !")


# In[17]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
print("+--------------------------------------+")
print("|  Which Algorithem you want to use    |")
print("+--------------------------------------+")
print("[1] LogisticRegression Score = ",lr.score(X_test,y_test))
print("[2] GaussianNB Score = ",gnb.score(X_test,y_test))
print("[3] SVC (Support Vectore Machine) Score = ",svm.score(X_test,y_test))
print("[4] KNeibhborsClassifier Score = ",knn.score(X_test,y_test))
print("[5] DecisionTreeClassifier Score = ",tree.score(X_test,y_test))


# In[28]:


def userUse(img,size):
    path=input("Enter image Path which only pixel (Path like 'D:\\image-classification\\') >> ")
    print("[*] reading image...")
    img=imread(path,as_gray=True)
    print("[+] Done !")
    print("[*] reducing your image size...")
    r_img=rescale(img,size)
    print("[+] Done !")
    print("[*] reshape your image into ",img.shape[0]*img.shape[1],"...")
    rs_img=np.reshape(r_img,r_img.shape[0]*r_img.shape[1])
    print("[+] Done !\n")
    return(rs_img)
def display_predict(predict):
    print("+----------------------------------+")
    print("|> This look like ",predict[0]," <|")
    print("+----------------------------------+\n")
    
    userWant=input("Do you want to predict other image (y OR n ) ? ").lower()
    if(userWant=='y'):
        Lets_Predict()
    else:
        print("+------------------------------------------+")
        print("|> Thanks for comming, hope so have enjoy <|")
        print("+------------------------------------------+")
        exit(0)
    
def Lets_Predict():
    algo=int(input("Which Algorithem you want to use ? "))

    if(algo==1):
        image=userUse(img,size)
        print("[*] Predicting your image ...")
        predict=lr.predict([image])
        display_predict(predict)

    elif(algo==2):
        image=userUse(img,size)
        print("[*] Predicting your image ...")
        predict=gnb.predict([image])
        display_predict(predict)

    elif(algo==3):
        image=userUse(img,size)
        print("[*] Predicting your image ...")
        predict=svm.predict([image])
        display_predict(predict)

    elif(algo==4):
        image=userUse(img,size)
        print("[*] Predicting your image ...")
        predict=knn.predict([image])
        display_predict(predict)

    elif(algo==5):
        image=userUse(img,size)
        print("[*] Predicting your image ...")
        predict=tree.predict([image])
        display_predict(predict)
        
Lets_Predict()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




