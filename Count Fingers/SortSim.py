
'''
Ajithesh Navaneethakrishnan
29th September, 2019
Comparing Quick, Heap & MergeSort based on their O(nlogn).
'''


#import libraries required.
import time
import sys
from heap import MaxHeap

integer_list = []

#read the file, store it in a array.
try:
    with open(sys.argv[1], 'r') as f:
        variable = f.read()
    #function to split and create float array.
    string_list = variable.split(";")
    for i in string_list:
        integer_list.append(int(i))

except:
    pass

#calc length of array
n = len(integer_list)

#QuickSort Implementation
def QuickSort(val):
    #define small, equal & greater arrays
    sml = []
    eql = []
    gtr = []

    if len(val) > 1:
        #calc pivot value
        piv = val[0]
        for x in val:
            if x < piv:
                sml.append(x)
            elif x == piv:
                eql.append(x)
            elif x > piv:
                gtr.append(x)
        #perform QuickSort for every value in array
        return QuickSort(sml)+eql+QuickSort(gtr)
    else:
        #return array if there's only one element
        return val

#HeapSort Implementation
def HeapSort(val):

    #define variables
    arr=[]
    n = len(val)
    #build maxheap
    mh = MaxHeap(val)
    #for every element in list do swap
    for  i in range(n-1,-1,-1):
        mh._data[i],mh._data[0] = mh._data[0],mh._data[i]
        #append array after sort
        arr.insert(0,mh._data[-1])
        #remove maxheap eat each step
        del mh._data[-1]
        #until last element in array call max_heapify
        mh.max_heapify(0,i)

    return arr

#MergeSort Implementation
def MergeSort(val):
    
    #perform split
    if len(val) > 1:
        mid = len(val) // 2
        L = val[:mid]
        R = val[mid:]

        #perform MergeSort on L & R
        MergeSort(L)
        MergeSort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                val[k] = L[i]
                i += 1
            else:
                val[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            val[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            val[k] = R[j]
            j += 1
            k += 1
    
    return val

#calc the start time
t1 = time.time()
QuickSort(integer_list)
#calc the end time
t2 = time.time()
#calc the time for the algorithm
tot = t2-t1
runtime_0 = '{0:.10f}'.format(tot)

#calc the start time
t1 = time.time()
HeapSort(integer_list)
#calc the end time
t2 = time.time()
#calc the time for the algorithm
tot = t2-t1
runtime_1 = '{0:.10f}'.format(tot)

#calc the start time
t1 = time.time()
MergeSort(integer_list)
#calc the end time
t2 = time.time()
#calc the time for algorithm
tot = t2-t1
runtime_2 = '{0:.10f}'.format(tot)


#write the files according to the logic
f = open("simulation.out", "w+")
#define every logic possible
if runtime_0 < runtime_1<runtime_2 or runtime_0<runtime_1==runtime_2:
    f.write('Quick'+ " "+ runtime_0 + "\n")
    f.write('Heap'+ " "+ runtime_1 + "\n")
    f.write('Merge'+ " "+ runtime_2)
elif runtime_0 < runtime_2<runtime_1:
    f.write('Quick'+ " "+ runtime_0 + "\n")
    f.write('Merge'+ " "+ runtime_2 + "\n")
    f.write('Heap'+ " "+ runtime_1)
elif runtime_1 < runtime_0 < runtime_2 or runtime_1==runtime_0<runtime_2:
    f.write('Heap'+ " "+ runtime_1 + "\n")
    f.write('Quick'+ " "+ runtime_0 + "\n")
    f.write('Merge'+ " "+ runtime_2)
elif runtime_1<runtime_2 <runtime_0 or runtime_1<runtime_0==runtime_2:
    f.write('Heap'+ " "+ runtime_1 + "\n")
    f.write('Merge'+ " "+ runtime_2 + "\n")
    f.write('Quick'+ " "+ runtime_0)
elif runtime_0==runtime_1==runtime_2:
    f.write('Heap'+ " "+ runtime_1 + "\n")
    f.write('Merge'+ " "+ runtime_2 + "\n")
    f.write('Quick'+ " "+ runtime_0)   
elif runtime_2<runtime_1 <runtime_0 or runtime_2<runtime_1==runtime_0:
    f.write('Merge'+ " "+ runtime_2 + "\n")
    f.write('Heap'+ " "+ runtime_1 + "\n")
    f.write('Quick'+ " "+ runtime_0)
elif runtime_2<runtime_0<runtime_1 or runtime_2==runtime_0<runtime_1:
    f.write('Merge'+ " "+ runtime_2 + "\n")
    f.write('Quick'+ " "+ runtime_0 + "\n")
    f.write('Heap'+ " "+ runtime_1)

#close the file
f.close()

#References