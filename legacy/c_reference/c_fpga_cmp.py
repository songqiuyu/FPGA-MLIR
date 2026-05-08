

import numpy as np

print('fpga result len='+str(80*80*64+40*40*64+20*20*64))

def arcmp(ar1,ar2):
    if(len(ar1)!=len(ar2)):
        print('for different len')
        return 1234
    i=0
    for value1,value2 in zip(ar1,ar2):
        if(value1!=value2):
            print('for different value:'+str(i)+' '+str(value1)+' '+str(value2))
            return 'diff'
        i+=1
    print('is the same')
    return 0



# get c results
c_node269=np.fromfile('./node269_57x80x80.image',dtype=np.int8).flatten()
c_node307=np.fromfile('./node307_57x40x40.image',dtype=np.int8).flatten()
c_node345=np.fromfile('./node345_57x20x20.image',dtype=np.int8).flatten()

# get fpga results
fpga_results=np.fromfile('./fpga_output.bin',dtype=np.int8).reshape(80*80*64+40*40*64+20*20*64)
fpga_node269=fpga_results[:80*80*64].reshape(80,80,64).transpose(2,0,1)[:57,:,:].flatten()
fpga_node307=fpga_results[80*80*64:80*80*64+40*40*64].reshape(40,40,64).transpose(2,0,1)[:57,:,:].flatten()
fpga_node345=fpga_results[80*80*64+40*40*64:80*80*64+40*40*64+20*20*64].reshape(20,20,64).transpose(2,0,1)[:57,:,:].flatten()

# now compare
arcmp(c_node269,fpga_node269)
arcmp(c_node307,fpga_node307)
arcmp(c_node345,fpga_node345)