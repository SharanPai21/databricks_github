# n = 7789

# dic = dict()

# s = str(n)

# for i in s:
#     if i in dic:
#         dic[i] += 1
#     else:
#         dic[i] = 1

# print(dic)


# n = 10400
# l = len(str(n))
# rev = 0
# while n>0:
#     d = n%10
#     rev = rev*10+d
#     n = n//10
# print(rev)

# n = 123
# n2 = n
# rev = 0
# d = 0
# while n > 0:
#     d = n%10
#     rev = rev*10+d
#     n = n//10

# print("palindrome") if rev == n2 else print("not")

# n = 1634
# n2 = n
# d = 0
# arm = 0
# l = len(str(n))
# while n > 0:
#     d = n%10
#     arm += d**l
#     n = n//10

# print("armstrong") if n2 == arm else print("not")


# n = 36

# fact = []

# for i in range(1, int(n**0.5), 1) :
#     if n%i == 0:
#         fact.append(i)
#     if n/i != i:
#         fact.append(int(n/i))

# fact.sort()
# print(" ".join(str(i) for i in fact))
# print(*fact, end = " ")

# n = 18
# sqrt = int(n**0.5)+1
# if n <= 2:
#     print("not prime")
# else:
#     for i in range(2, sqrt):
#         if n%i == 0:
#             print("not prime")
#             break
#     else:
#         print("prime")   


n1 = 40
n2 = 20

def gcd(n1, n2):
    for i in range(min(n1,n2), 0, -1):
        if n1 % i == 0 and n2 % i == 0:
            gcd = i 
            break
    print(gcd)

gcd(n1,n2)
gcd(n1-n2, n2)