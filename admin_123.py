# -*- coding: cp936 -*-
name = raw_input("�˻���\n")
password = raw_input("����\n")
isOK = 0

for count in range(1, 3):
    if name == "admin" and password == "123":
        print("��½�ɹ�")
        isOK = 1
        break
    else:
        print("�˺Ż��������\n")
        name = raw_input("�˻���\n")
        password = raw_input("����\n")

if isOK != 1:
    print("��½ʧ��")
