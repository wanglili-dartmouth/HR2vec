with open('out.slashdot-threads') as f:
    array=[]
    for line in f: 
        array.append([int(x) for x in line.split()])
print(array)
with open('edgelist.tsv',"w")as f:
    for a in array:
        #print(a)
        f.write(str(a[0])+'	'+str(a[1])+'	'+str(int(a[3]/3600/24/30)+1)+'\n')
       
        