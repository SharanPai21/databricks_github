# ---------- Tuple ----------

tup = (1, 2, 3, 4)

print(tup.count(1))

# Unpacking values
first, *middle, last = tup
print(first, middle, last)

# Ignoring values
first, *_, last = tup
print(first, _, last)

print(1 in tup)


# ---------- Strings (immutable example) ----------

# h = "hello"
# h[0] = "g"     # This would raise an error
# print(h[0])


# ---------- List ----------

lst = [1, 2, 3, 4, 5, 6, 7]

lst.append(8)
print(lst)

lst.insert(9, 9)
print(lst)

lst.remove(1)
print(lst)

popped = lst.pop()
print(popped)
print(lst)

print(lst[::-1][0])   # Last element
print(lst[:])         # Full copy
print(lst[:7])
print(max(lst))


# ---------- Repeating list ----------

lst1 = ["black tea", "sugar"] * 3
print(lst1)


# ---------- Bytearray ----------

byte_data = bytearray(b"CARDAMOM")
print(byte_data)

byte_data = byte_data.replace(b"CARD", b"BOARD")
print(byte_data)


# ---------- Sorting ----------

# sort() modifies the list in place and returns None
print(lst.sort())
print(lst)


# ---------- Sets ----------

set_a = {"a", "b", "c", "d"}
set_b = {"c", "d", "e", "f"}

union = sorted(set_a | set_b)
print(union)

print("a" in union)

print(set_a.difference(set_b))


# ---------- Dictionaries ----------

chai = {"order": "tea", "type": "masala", "sugar": "0"}
print(chai)

chai["milk"] = 5
print(chai["sugar"])

del chai["milk"]

print(chai.popitem())

chai_extra = {"spice": "cardamon", "spice_type": "crushed"}
chai.update(chai_extra)
print(chai)

# Safe access
customer_note = chai.get("customer_note", "not found")
print(customer_note)


# ---------- Input ----------

#user_input = input("input: ")
#print(type(user_input))


# ---------- Loop with tuple unpacking ----------

staff = [("ravi", 19), ("raj", 16), ("trophy", 18)]

for name, age in staff:
    if age > 2:
        print(name, age)
        continue


# ---------- walrus ----------

value = 13
rem = value % 5

if rem:
    print("not divisible")

#with walrus operator

if reminder := value % 2:
    print("not divisible")

size = ["big", "medium", "small"]

# if (sizee := input("give the size")) in size:
#     print(f"ordered a {sizee} pizzeee")
# else:
#     print("enter valid size(big/medium/small)")

# while (s := input("enter size")) not in size:
#     print(f"{s} is not available")
# print(f"ordered {s}")


# dictionary case 

users = [
    {"id": 1, "total": 100, "coupon": "P20"},
    {"id": 2, "total": 150, "coupon": "F10"},
    {"id": 3, "total": 80, "coupon": "P50"},
]

discounts = {
    "P20": (0.2, 0),
    "F10": (0.5, 0),
    "P50": (0, 10),
}

for user in users:
    percent, flat = discounts.get(user['coupon'], (0, 0))
    discount = user['total'] * percent + flat
    print(f"user{user['id']} paid {user['total']} and got discount of {discount}")



def print_order():
    chai_order = "a"
    print("outer", chai_order)
    def chai_order_2():
        chai_order = "b"
        print("innner", chai_order)
    
    chai_order_2()

print_order()


def type1():
    chai_type = "plain"
    def type2():
        nonlocal chai_type
        chai_type = "ginger"

    type2()
    print(chai_type)

type1()


chai_type = "masala"
def type3():

    def type4():
        global chai_type
        chai_type = "irani"

    type4()

type3()
print(chai_type)


chai =[1,2,3,4,5,6,7]

def edit_chai(c):
    c[1] = 69

edit_chai(chai)
print(chai)


def func_list(*g, **gg):
    print(g)
    print(gg)

func_list(1,3,2,4,5,6,7, sugar = 'high', milk = 'medium')

# def test(order = []):
#     order.append('masala')
#     print(order)

# test([1,2,3,4,5])
# test()
# test()
# print(order)

def test(order = None):
    if order is None:
        order = []

    print(order)

test(['a','b','b'])
test([1,2,3,4])


def generate_invoice(customer_name: str = "Guest", *items: str, **charges: float) -> str:
    # Write your code below this line
    total = 0.0
    invoice_lines = [f"Invoice for {customer_name}:"]
 
    if items:
        invoice_lines.append("Items:")
        for item in items:
            invoice_lines.append(f"- {item}")
 
    if charges:
        invoice_lines.append("Charges:")
        for label, amount in charges.items():
            invoice_lines.append(f"{label.capitalize()}: {amount}")
            total += amount
 
    invoice_lines.append(f"Total Amount Due: {total}")
    return "\\n".join(invoice_lines)


def generate_invoice(customer_name: str = 'Guest', *items: str, **charges: float) -> str:
    t = 0
    itemss = [f"Invoice for {customer_name}"]
        
    if items:
        itemss.append("Items:")
        for i in items:
            itemss.append(f"- {i}")
            
    if charges:
        itemss.append("Charges:")
        for i, j in charges.items():
            itemss.append(f"{i.capitalize()}: {j}")
            t += j
               
    itemss.append(f"Total Amount Due: {t}")
    return "\\n".join(itemss)    


# def idel_chaiwala():
#     pass

# print(idel_chaiwala())

def rem(cups):
    if cups == 0:
        return "no chai"
    return "chai is available"

g = rem(0)
print(g)        

def mul():
    return 100, 200, 300, 400, 500, 600

a, b, *c = mul()

print(a,b, c)

def pure_chai(cups):
    return cups*10

total = 0

def impure_chai(cups):
    global total
    total = cups*10

def rec(num):
    if num == 0:
        return 1
    elif num == 1:
        return 1
    else:
        print(num, end = " ")
        small = rec(num-1)
        return  num*small
    
print("\n",rec(1))

def chai(n):
    if n == 0:
        return "nope"
    return chai(n-1)
    
print(chai(10))


numbers = [1,2,3,4,5]
sq = list(map(lambda x: x**2, numbers))
print(sq)

db = list(map(lambda a: a*2, numbers))
print(db)

evenodd = list(filter(lambda x: x%2 == 0, numbers))
print(evenodd)

double = lambda x: x*2
d = double(69)
print(d)

lst = [1,1,1,2,3,3,3,4,4,4,5]

f = list(filter(lambda x: x != 4, numbers))
print(f)