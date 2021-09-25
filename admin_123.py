# -*- coding: cp936 -*-
name = input("账户名\n")
password = input("密码\n")
isOK = 0

for count in range(1, 3):
    if name == "admin" and password == "123":
        print("登陆成功")
        isOK = 1
        break
    else:
        print("账号或密码错误\n")
        name = input("账户名\n")
        password = input("密码\n")

if isOK != 1:
    print("登陆失败")
