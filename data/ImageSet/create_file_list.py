
file_a = open("allV2.txt","w")
file_t=open("trainV2.txt","w")
file_v=open("valV2.txt","w")
for i in range(7481):
    file_a.write("{0:06d}\n".format(i))
    if i<1870:
        file_v.write("{0:06d}\n".format(i))
    else:
        file_t.write("{0:06d}\n".format(i))
        
file_a.close()
file_t.close()
file_v.close()