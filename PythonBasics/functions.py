# def db_call():
#     print("MongoDB Connection Soon...!")
# db_call()


# def db_call(username,password):
#     #print("Login Success") if username == "admin" and password == "admin@123" else print("Login Fail")
#     if username == "admin" and password == "admin@123":
#         print("Login Success")
#     else:
#         print("Login Fail")
# db_call("admin","admin@123")


# def db_call():
#     return "DB Connection Soon"
# res = db_call()
# print(res)

# res=""
# def db_call(username,password):
#     if username == "admin" and password == "admin@123":
#         res = "Login Success"
#     else:
#         res = "Login Fail"
#     return res
# x = db_call("admin","admin@123")
# print(x)


# def greet(msg="Good Evening !!!"):
#     print("Hello",msg)

# greet()
# greet("Good Morning")


# def add(*numbers):
#     print(sum(numbers))

# add(10,20,30)

# def square(num):
#     return num*num
# res = square(10)
# print(res)

# res = lambda num: num*num
# print(res(10))

# def outer():
#     def inner():
#         print("Hello, inner function")
#     inner()
# outer()

add = lambda a,b : a+b
print(add(200,100))



