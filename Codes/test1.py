from jtop import jtop

def main():
    while(True):    
        jetson = jtop()
        jetson.start()
        x=jetson.power
        jetson.close()
        print(x[0])

if __name__ == "__main__":
    main()