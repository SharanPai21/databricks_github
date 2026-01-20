# # -------------------------
# # 1. Square of stars
# # -------------------------
# for i in range(4):
#     for j in range(4):
#         print("*", end=" ")
#     print()


# # -------------------------
# # 2. Number triangle (1, 1 2, 1 2 3 ...)
# # -------------------------
# for i in range(4):
#     for j in range(i + 1):
#         print(j + 1, end=" ")
#     print()


# # -------------------------
# # 3. Repeated number triangle (1, 2 2, 3 3 3 ...)
# # -------------------------
# for i in range(4):
#     for j in range(i + 1):
#         print(i + 1, end=" ")
#     print()


# # -------------------------
# # 4. Inverted star triangle
# # -------------------------
# for i in range(4, 0, -1):
#     for j in range(i):
#         print("*", end=" ")
#     print()


# # -------------------------
# # 5. Pyramid
# # -------------------------
# def print_pyramid(height):
#     for i in range(height):
#         for j in range(height - i - 1):
#             print(" ", end=" ")
#         for j in range(2 * i + 1):
#             print("*", end=" ")
#         print()


# # -------------------------
# # 6. Inverted pyramid
# # -------------------------
# def print_inverted_pyramid(height):
#     for i in range(height, 0, -1):
#         for j in range(height - i):
#             print(" ", end=" ")
#         for j in range(2 * i - 1):
#             print("*", end=" ")
#         print()


# print_pyramid(5)
# print_inverted_pyramid(5)


# # -------------------------
# # 7. Number pattern (up and down)
# # -------------------------
# for i in range(5):
#     for j in range(i):
#         print(j, end=" ")
#     print()

# for i in range(5, 0, -1):
#     for j in range(i):
#         print(j, end=" ")
#     print()


# # -------------------------
# # 8. Diamond
# # -------------------------
# size = 5

# for i in range(size):
#     for j in range(size - i - 1):
#         print(" ", end=" ")
#     for j in range(2 * i + 1):
#         print("*", end=" ")
#     print()

# for i in range(size - 1, 0, -1):
#     for j in range(size - i):
#         print(" ", end=" ")
#     for j in range(2 * i - 1):
#         print("*", end=" ")
#     print()


# # -------------------------
# # 9. Star triangle (up and down)
# # -------------------------

# def n(n):
#     for i in range(n*2):
#         stars = i
#         if i > n:
#             stars = 2*n-i
#         for j in range(stars):
#             print("*", end = " ")
#         print()

# n(15)

# start = 1
# for i in range(1, 6, 1):
#     start = 1 if i % 2 == 0 else 0
#     for j in range(i):
#         start = 1 - start  
#         print(start, end = " ")
#     print()


# for i in range(1, 6, 1):
#     for j in range(i):
#         print(f"{j+1}", end = " ")
#     for j in range(2*5 - 2*i):
#         print(" ", end = " ")
#     for j in range(i, 0, -1):
#         print(f"{j}", end = " ")         
#     print()

# count = 0
# for i in range(5):
#     for  j in range(i+1):
#         count += 1
#         print(count, end = " ")
#     print()

# for i in range(5):
#     for j in range(i+1):
#         print(chr(65+j), end = " ")
#     print()

# for i in range(1, 6, 1):
#     for j in range(i):
#         print(chr(64+i), end = " ")
#     print()

# for i in range(5, 0, -1):
#     for j in range(i):
#         print(chr(65+j), end = " ")
#     print()

# for i in range(5):
#     for j in range(i):
#         print(" ", end = " ")
#     for j in range(5-i):
#         print(chr(65+j), end = " ")
#     print()


# for i in range(5):
#     c = 65
#     br = int((2*i+1)/2)
#     for j in range(5-i):
#         print(" ", end = " ")
#     for j in range(2*i+1):
#         print(chr(c), end = " ")
#         c += 1 if j < br else -1
#     print()

# def n(n):
#     for i in range(n):
#         c = 64+n - i
#         for j in range(i+1):
#             print(chr(c), end = " ")
#             c += 1
#         print()

# n(5)
# --------------------------------------------
# for i in range(5): 
#     for j in range(5 - i):
#         print("*", end = " ")
#     for j in range(2*i):
#         print(" ", end = " ")
#     for j in range(5 - i):
#         print("*", end = " ")
#     print()
# for i in range(3, -1, -1): 
#     for j in range(5 - i):
#         print("*", end = " ")
#     for j in range(2*i):
#         print(" ", end = " ")
#     for j in range(5 - i):
#         print("*", end = " ")
#     print()

# spaces = 2*5 - 2
# for i in range(1, 2*5, 1):
#     stars = i 
#     if i > 5:
#         stars = 2*5 - i
#     for j in range(stars):
#         print("*", end = " ")
#     for j in range(spaces):
#         print(" ", end = " ")
#     for j in range(stars):
#         print("*", end = " ")
#     if i < 5:
#         spaces -= 2
#     else:
#         spaces += 2
#     print()

# for i in range(5):
#     for j in range(5):
#         if i == 0 or j == 0 or i == 4 or j == 4:
#             print("*", end = " ")
#         else:
#             print(" ", end = " ")
#     print()

n = 6
for i in range(2*n-1):
    for j in range(2*n-1):
        top = i
        left = j
        right = (2*n-1) - 1 - j
        bottom = (2*n-1) - 1 - i
        print(n - min(min(top, bottom), min(left, right)), end = " ")
    print()