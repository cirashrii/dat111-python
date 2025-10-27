

#funksjoner & lister
# Her med typer spesifisert
def kvadrer(n:int) -> int:
    return n*n

list0 = [1,2,3,14,5,6]
list1 = []

for x in list0:
    list1.append(kvadrer(x))
#alternativt
print(list0)
print(list1)



