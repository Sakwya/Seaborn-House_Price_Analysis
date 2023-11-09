import openpyxl

f = open("info.txt", encoding="utf-8", mode="r+")
lines = f.readlines()
s = False
info_list = []
info = []
for i in range(len(lines)):
    line = lines[i]
    if s:
        if ':' not in line:
            info.append(line)
            continue
        else:
            s = False
            info_list.append(info)
    info = []
    if ':' in line:
        info = line.split(': ')
        if len(info) < 2:
            s = True
        else:
            info_list.append(info)

workbook = openpyxl.Workbook()
sheet = workbook.active
for i in range(len(info_list)):
    info = info_list[i]
    line = []
    if len(info) > 2:
        line.append(info[0].rstrip('\n'))
        s = ''
        for i in range(len(info)):
            if i > 0:
                s = s + info[i]
        line.append(s.rstrip('\n'))
    else:
        line.append(info[0].rstrip('\n'))
        line.append(info[1].rstrip('\n'))
    sheet.append(line)
workbook.save("info.xlsx")
