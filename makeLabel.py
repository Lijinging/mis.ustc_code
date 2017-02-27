import os

x = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
f = open("label.txt", "w")
for i in x:
    f.write(i+"\n")
f.close()