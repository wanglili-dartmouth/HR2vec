with open('email-Eu-core-temporal (1).txt') as f:
    array=[]
    for line in f: 
        array.append([int(x) for x in line.split()])
print(array)
with open('edge.tsv',"w")as f:
    for a in array:
        #print(a)
        f.write(str(a[0])+'	'+str(a[1])+'	'+str(int(a[2]/3600/24)+1)+'\n')
       
        