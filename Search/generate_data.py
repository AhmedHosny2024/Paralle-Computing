# write 1000000 ones each 1 in line in a file
file=open('data.txt','w')
for i in range(1000000):
    file.write('1\n')
file.close()

