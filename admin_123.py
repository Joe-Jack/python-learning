# -*- coding: cp936 -*-
name = input("�˻���\n")
password = input("����\n")
isOK = 0

for count in range(1, 3):
    if name == "admin" and password == "123":
        print("��½�ɹ�")
        isOK = 1
        break
    else:
        print("�˺Ż��������\n")
        name = input("�˻���\n")
        password = input("����\n")

if isOK != 1:
    print("��½ʧ��")
