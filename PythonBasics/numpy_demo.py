import numpy as np

marks = np.array([10,50,70,100])
result = np.where(marks>50, "pass","fail")
print(result)

# arr1 = np.array([[1,2],[3,4]])
# print( arr1.sum(axis=0) )
# print( arr1.sum(axis=1) )

# arr1 = np.array([10,50,20,80,30])
# print(arr1.argmax())
# print(arr1.argmin())


# arr1 = np.array([[1,2],[3,4]])
# flat = arr1.flatten()
# print(flat)


# arr1 = np.array([1,2,3])
# print(arr1.shape) #(3,) 1-D
# arr2 = np.array([[1,2,3],[4,5,6]])
# print(arr2.shape) #(2, 3) 2-D

# marks = np.array([30,40,50,60,70,80,90,100])
# marks[marks<50] = 0
# print(marks)


# marks = np.array([30,40,50,60,70,80,90,100])
# passed = marks[marks>50]
# print(passed)


# arr1 = np.array([10,20,30,40])
# arr2 = arr1[1:3]    #[20,30]
# arr2[0] = 999       #[999,30]
# print(arr2)
# print(arr1)   # [10,999,30,40]



# arr = np.array([1,2,3,4,5])
# e1 = arr[0]
# print(e1)
# e2 = arr[1]
# print(e2)
# e5 = arr[4]
# print(e5)

# arr1 = np.array([[1,2,3],
#                  [4,5,6]])
# print(arr1)
# e1 = arr1[0][0]
# print(e1)
# e2 = arr1[1][2]
# print(e2)
# print(arr1.shape)

# arr = np.array([10,20,30,40,50])
# print("Sum :",np.sum(arr))
# print("MEAN :", np.mean(arr))
# print("Max :", np.max(arr))
# print("Min :",np.min(arr))


# arr1 = np.array([1,2,3])
# arr2 = np.array([4,5,6])
# res = arr1 + arr2
# print(res)


# arr1 = np.array([[1,2],
#                  [3,4]])
# print(arr1.shape)

# arr1 = np.array([1,2,3,4,5,6])
# res = arr1.reshape(2,3)
# print(res)

# arr1 = np.array([1,2,3,4,5,6])
# even = arr1[arr1 % 2 == 0]
# print(even)
# odd = arr1[arr1 % 2 != 0]
# print(odd)
# print(arr1.size)
# print(even.size)
# print(odd.size)

arr1 = np.array([10,50,20,40,30])
res = np.sort(arr1)
#print(res)



# arr = np.zeros(10)
# print(arr)

# arr1 = np.ones(5)
# print(arr1)

# arr2 = np.arange(1,10)
# print(arr2)

# arr3 = np.arange(1,10,2)
# print(arr3)

# import numpy as np
# arr = np.array([10,20,30,40,50]);
# #Addition
# print(arr+10)
# #Subtraction
# print(arr-5)


# import numpy as np
# arr = np.array([10,20,30,40,50])
# print(arr+10)



# import numpy as np
# arr = np.array([10, 20, 30, 40, 50])
# print(arr)
