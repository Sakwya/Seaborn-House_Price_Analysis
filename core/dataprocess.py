import csv

f = open("../house_info.csv", mode='r', newline='', encoding="utf-8-sig")
dr = csv.DictReader(f)
n_f = open("house_info.csv", mode='w', newline='', encoding="utf-8-sig")
dw = csv.DictWriter(n_f, fieldnames=dr.fieldnames)
dw.writeheader()
dw.writerows(dr)
n_f.close()
f.close()
