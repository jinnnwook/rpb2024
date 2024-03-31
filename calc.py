def main():
    print("Let's implement division. Type two numbers for x and y")
    
    x = int(input("x > "))
    y = int(input("y > "))
    if y==0:
        print('Error: cannot divide by zero.')
    else:
        print("%d / %d = %0.3f" % (x, y, divide(x, y)))
    
    
def add(a, b):
    return a + b
    
    
# TODO: divide() 함수 정의
def divide(x,y):
    if y==0:
        print('Error: cannot divide by zero.')
    else:
        return x/y

if __name__ == "__main__":
    main()
