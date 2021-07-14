#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("+==============================================================+")
print("|>                                                            <|")
print("|>      Application Name     :    Loan Pridiction             <|")
print("|>      Developer            :    Satya Mishra                <|")
print("|>      Investigator         :    Mr. Pankaj Shukla           <|")
print("|>      Email                :    satyamishra559@gmail.com    <|")
print("|>      Language Used        :    Python 3 and Python 2       <|")
print("|>      Based On             :    Machine Learning            <|")
print("|>                                                            <|")
print("+==============================================================+")

def load(msg):
    print("[*]",msg,"...")
    
def result(msg):
    print("[+]",msg)

def done():
    print("[+] Done !")
    
def printLine():
    print("+------------------------------------------------------------------+")

def userInput(msg):
    value=input(msg)
    return value

def lenght(lst):
    j=0
    for i in lst:
        j+=1
    return j
    
load("Loading all required module")
import pandas as pd 
import numpy as np
done()

load("Loading all Machine Learning Algorithem")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skimage.io import imread,imshow
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 
load("It take some time, please wait")
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
done()


# In[2]:


# Reading data from csv file
file=input("\nEnter csv file path for training Machine (D:\\\path\\\ filename.csv) >")
load("Reading file")
csvFile=pd.read_csv(file)
done()


# In[3]:


col=csvFile.columns
j=1
print("------------------- For Removing unecessary data -------------------")
for i in col:
    print("[",j,"]",i)
    j+=1
printLine()

delCol=input("Which columns you want to remove Ex: (1,2,3 or 2) >> ").split(",")
delColumns=[]
for i in delCol:
    delColumns.append(int(i)-1)
y=int(input("Which columns Outcome(Result) data Ex: (10) >> "))
y=y-1
printLine()
load("Please wait whlile second")
j=0
for i in delColumns:
    csvFile.drop([csvFile.columns[j]],axis=1,inplace=True)
    j+=1
X=csvFile.drop([csvFile.columns[y-j]],axis=1)
org_feature=X
y=csvFile.iloc[:,y-j]
# y=pd.DataFrame(y,columns=["Loan_status"])


# In[4]:


load("Data preprocessing, Checking null value")


# In[8]:


def myFun(lst):
    usrPut=input("Enter value which is Maximume >> ").title().strip()
    if(usrPut in lst):
        return usrPut
    else:
        print("[-] Opps ! invalid input..")
        return False
j=0
total_null=0
for i in X:
    single_col=X[X.columns[j]].isnull().sum()
    total_null+=single_col
    j+=1
if(total_null==0):
    result("Don't have any null value, that's good :)")
else:
    j=0
    print("[*] You have",total_null,"Null data, we are going to make good data...")
    
    def fillData(value):
        load("Filling")
        X[i].fillna(value,inplace=True)
        done() 

    def fillFunction():
        dataType=int(input("Enter dataType which you want to fill ? "))

        if(dataType==1):
            value=input("Enter String >> ").title()
            fillData(value)

        elif(dataType==2):
            value=int(userInput("Enter Integer value >> "))
            fillData(value)

        elif(dataType==3):
            value=float(userInput("Enter Float value >> "))
            fillData(value)

        elif(dataType==4):
            value=X[i].mean()
            fillData(value)

        else:
            print("[-] Warning ! Please input valid value.")
            fillFunction()  
    for i in X:
        single_col=X[X.columns[j]].isnull().sum()
        if(single_col != 0):
            print("\n\n|>--------- ",i,"Have ---------<|")
            print(X[i].value_counts())
            print("|>-----------------------------<|")
            print("[1] To Fill String value EX:('hello'))")
            print("[2] To Fill Integer value EX:(1,3,4..etc)")
            print("[3] To Fill float value EX:(1.0,4.3 somthing)")
            print("[4] To Fill value by avarage EX:(Add all value/100 )")
            fillFunction()
        j+=1


# In[9]:


print("\n[+] Your data successfully cleaned...\n")


# In[10]:


load("Creating your data into Machine readable formate")
X=pd.get_dummies(X)
done()


# In[11]:


print("")
load("Normalizing data")
scale=MinMaxScaler(feature_range=(0,1))
X[X.columns]=scale.fit_transform(X[X.columns])
done()
print("")


# In[12]:


split_size=int(input("Enter size to spliting Testing data recommand(25) >> "))/100
load("Spliting your data into Training and Testing")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=split_size)
done()


# In[13]:


result("You have to done maximume 90% work !")


# In[14]:


load("Fit Training Data into Machine,it take some time")
lr=LogisticRegression()
lr.fit(X_train,y_train)
print("Loadining.. 10%")

svm=SVC()
svm.fit(X_train,y_train)
print("Loadining......... 25%")

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
print("Loadining................ 45%")

gnb=GaussianNB()
gnb.fit(X_train,y_train)
print("Loadining........................ 75%")

mnb=MultinomialNB()
mnb.fit(X_train,y_train)
print("Loadining............................... 80%")

tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)
print("Loadining....................................... 100%")


# In[15]:


print("")
load("Predicting score of model")


# In[16]:


lr_score=lr.score(X_test,y_test)
svm_score=svm.score(X_test,y_test)
knn_score=knn.score(X_test,y_test)
gnb_score=gnb.score(X_test,y_test)
mnb_score=mnb.score(X_test,y_test)
tree_score=tree.score(X_test,y_test)
done()
result("Model is ready for use !")


# In[ ]:





# In[32]:


while(True):
    def wantAgain():
        want=input("Do you want to predict any other Client Loan (y OR n) ? ").lower().strip()
        if(want=="y"):
            return 1
        else:
            print("[+] Thanks for comming, hope so you are enjoy here...")
            return 0
        
    def algoError():
        algo_use=int(input("Which Algorithem you want to use (1-6) >> "))
        if(algo_use >= 1 and algo_use <= 6):
            return algo_use
        else:
            print("[-] Warning ! please input valid number 1-6 only >> ")
            return False
    
    print("+---------------------------------------------------------+")
    print("|1| LogisticRegression Score=",lr_score*100,"%")
    print("|2| SVM (Support Vector Machine) Score=",svm_score*100,"%")
    print("|3| KNeighborsClassifier Score=",knn_score*100,"%")
    print("|4| GaussianNB Score=",gnb_score*100,"%")
    print("|5| MutinomialNB Score=",mnb_score*100,"%")
    print("|6| DecisionTreeClassifier Score=",tree_score*100,"%")
    print("+---------------------------------------------------------+")

    algo_use=algoError()
    if(algo_use==False):
        algo_use=algoError()
        if(algo_use==False):
            algo_use=algoError()
            if(algo_use==False):
                algo_use=algoError()
                if(algo_use==False):
                    algo_use=algoError()
                    if(algo_use==False):
                        algo_use=algoError()
                        if(algo_use==False):
                            algo_use=algoError()
                            if(algo_use==False):
                                algo_use=algoError()
                                if(algo_use==False):
                                    print("[+] Error ! TimeOut please run again all code...")

    print("+---------------------------------------------------------+")


    # Implementing model------------------------------------------------
    name=input("|> Enter your name >> ").title()
    print("+---------------------------------------------------------+")

    # ------------------------------------------ user handle ---------------------------------------
    #Validation function declear
    def validateData(input_data,list):
        # Checking enterd data valid?
        if(input_data in list):
            return True
        else:
            print("[-] Oops ! Invalid input.")
            return False

    # Intereger validation
    def floatIntValidate(input_data,list):
        # Checking enterd data valid?
        if(input_data >= list[0] and input_data <= list[-1]):
            return True
        else:
            print("[-] Oops ! Invalid input. its only take under ("+str(list[0])+" to "+str(list[-1])+") value")
            return False


    userInputData=[]
    csvFile=org_feature
    j=0
    for i in csvFile:
        #If value is string  
        if(type(csvFile[csvFile.columns[j]][0]) == str):
            list=[]     
            # Filtering columns
            for k in csvFile[csvFile.columns[j]]:
                if(k in list):
                    continue
                elif(type(k)==str):
                    list.append(k)
            #Inputting data from user
            if(lenght(list) <= 2):   
                print("")
                input_data=input(name+"'s "+csvFile.columns[j]+" ("+str(list[0])+" OR "+str(list[-1])+") >> ").title()
                # Checking enterd data valid?
                if(validateData(input_data,list)==False):
                    input_data=input(name+"'s "+csvFile.columns[j]+" ("+str(list[0])+" OR "+str(list[-1])+") >> ").title()
                    input_data=input_data.strip()
                    if(validateData(input_data,list)==False):
                        input_data=input(name+"'s "+csvFile.columns[j]+" ("+str(list[0])+" OR "+str(list[-1])+") >> ").title()
                        input_data=input_data.strip()
                        if(validateData(input_data,list)==False):
                            input_data=input(name+"'s "+csvFile.columns[j]+" ("+str(list[0])+" OR "+str(list[-1])+") >> ").title()
                            input_data=input_data.strip()
                            if(validateData(input_data,list)==False):
                                print("[-] Opps ! Time out, you have to restart all programm.")
                                break      
            else:
                print("")
                input_data=input(name+"'s Enter any "+csvFile.columns[j]+" "+str(list)+" >> ").title()
                input_data=input_data.strip()
                # Checking enterd data valid?
                if(validateData(input_data,list)==False):
                    input_data=input(name+"'s Enter any "+csvFile.columns[j]+" "+str(list)+" >> ").title()
                    input_data=input_data.strip()
                    if(validateData(input_data,list)==False):
                        input_data=input(name+"'s Enter any "+csvFile.columns[j]+" "+str(list)+" >> ").title()
                        input_data=input_data.strip()
                        if(validateData(input_data,list)==False):
                            input_data=input(name+"'s Enter any "+csvFile.columns[j]+" "+str(list)+" >> ").title()
                            input_data=input_data.strip()
                            if(validateData(input_data,list)==False):
                                print("[-] Opps ! Time out, you have to restart all programm.")
                                break
            userInputData.append(input_data)

        #If value is integer    
        elif(type(csvFile[csvFile.columns[j]][0]) == int):
            #  filtering cloumns       
            Intlist=[] 
            minVal=int(csvFile[csvFile.columns[j]].min())
            Intlist.append(minVal)
            maxVal=(csvFile[csvFile.columns[j]].max())
            Intlist.append(maxVal)

            print("")
            input_data=int(input("Enter "+csvFile.columns[j]+" value (Min : "+str(csvFile[csvFile.columns[j]].min())+", Max : "+str(csvFile[csvFile.columns[j]].max())+") >>"))
           # Checking enterd data valid?
            if(floatIntValidate(input_data,Intlist)==False):
                input_data=int(input("Enter "+csvFile.columns[j]+" value (Min : "+str(csvFile[csvFile.columns[j]].min())+", Max : "+str(csvFile[csvFile.columns[j]].max())+") >>"))
                if(floatIntValidate(input_data,Intlist)==False):
                    input_data=int(input("Enter "+csvFile.columns[j]+" value (Min : "+str(csvFile[csvFile.columns[j]].min())+", Max : "+str(csvFile[csvFile.columns[j]].max())+") >>"))
                    if(floatIntValidate(input_data,Intlist)==False):
                        input_data=int(input("Enter "+csvFile.columns[j]+" value (Min : "+str(csvFile[csvFile.columns[j]].min())+", Max : "+str(csvFile[csvFile.columns[j]].max())+") >>"))
                        if(floatIntValidate(input_data,Intlist)==False):
                            print("[-] Opps ! Time out, you have to restart all programm.")
                            break
            userInputData.append(input_data)

        #If value is Float  
        else:
            floatList=[] 
            flist1=float(csvFile[csvFile.columns[j]].min())
            floatList.append(flist1)
            flist2=float(csvFile[csvFile.columns[j]].max())
            floatList.append(flist2)

            print("")
            input_data=float(input("Input "+csvFile.columns[j]+" value (Min : "+str(csvFile[csvFile.columns[j]].min())+", Max : "+str(csvFile[csvFile.columns[j]].max())+") >>"))
            # Checking enterd data valid?
            if(floatIntValidate(input_data,floatList)==False):
                input_data=float(input("Input "+csvFile.columns[j]+" value (Min : "+str(csvFile[csvFile.columns[j]].min())+", Max : "+str(csvFile[csvFile.columns[j]].max())+") >>"))
                if(floatIntValidate(input_data,floatList)==False):
                    input_data=float(input("Input "+csvFile.columns[j]+" value (Min : "+str(csvFile[csvFile.columns[j]].min())+", Max : "+str(csvFile[csvFile.columns[j]].max())+") >>"))
                    if(floatIntValidate(input_data,floatList)==False):
                        input_data=float(input("Input "+csvFile.columns[j]+" value (Min : "+str(csvFile[csvFile.columns[j]].min())+", Max : "+str(csvFile[csvFile.columns[j]].max())+") >>"))
                        if(floatIntValidate(input_data,floatList)==False):
                            print("[-] Error (404) Time out, you have to restart all programm.")
                            break
            userInputData.append(input_data)
        j+=1


    # ------------------------------ user input's data preprocessing -------------------------------------------------
    load("please wait")
    userInput=pd.DataFrame([userInputData],columns=csvFile.columns)

    # Making machine readable formate
    load("Making your data into Machine readable formate")
    newDf=pd.get_dummies(userInput)
    done()

    load("Checking missing columns")
    missCol=set(X.columns)-set(userInput.columns)

    load("filling missing columns")
    for i in missCol:
        newDf[i]=0
    done()

    # Predict ----------------------------
    def WhichAlgo(algo,data):
        if(algo==1):
            return lr.predict(data)
        elif(algo==2):
            return svm.predict(data)
        elif(algo==3):
            return knn.predict(data)
        elif(algo==4):
            return gnb.predict(data)
        elif(algo==5):
            return mnb.predict(data)
        elif(algo==4):
            return tree.predict(data)

    print("")
    load("Predicting")
    predict=WhichAlgo(algo_use,newDf)

    # ---------------------Result / Outcome
    if(predict[0]=='Y'):
        print("")
        print("+=======================================================================+")
        print("|> Congratulation Mr/Mrs.",name ,"! Your loan has been approved ")
        print("+=======================================================================+\n")
    else:
        print("+===================================================================+")
        print("|> \tSorry ",name,"! Your loan is not approved . ")
        print("+===================================================================+\n")
        
    want=wantAgain()
    if(want==1):
        continue
    else:
        break


# In[ ]:




