datafile=open("gold_pred.csv","r")
godefile=open("gold_pred_new.csv","w")
gold_data=datafile.read()
gold_data=gold_data.splitlines()
day=1
for i in gold_data:
    rprice,pprice=i.split(",")
    godefile.write(rprice+','+pprice+'\n')
    if day==4:
        godefile.write('-1'+','+pprice+'\n')
        godefile.write('-1'+','+pprice+'\n')
        day=(day+2)%7    
    day=(day+1)%7



