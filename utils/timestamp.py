from datetime import datetime

def timestamp():
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    return str(int(ts))
    
def destamp(stamp):
    time = datetime.fromtimestamp(stamp)
    print(time)

if __name__ == '__main__':
    ts = timestamp()
    destamp(ts)