file = open("take5.mid", "rb")
for chunk in file:
    for byte in chunk:
        print(bin(byte))
print(file.read()[0:7])