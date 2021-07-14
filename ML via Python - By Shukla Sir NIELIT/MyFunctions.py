#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def banner():
    print("+==============================================================+")
    print("|>                                                            <|")
    print("|>      Application Name     :    SPAM Detection              <|")
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

