
class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.list = []

    def update(self, val, n=1):
        self.val = val
        self.list.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    loss_recoder = AvgMeter()
    loss_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    batch_size = 15
    for loss in loss_list:
        loss_recoder.update(loss, n = batch_size)
        print(loss_recoder.avg)
    print('-'*10)
    print(loss_recoder.avg)
    loss_recoder.reset()
    print(loss_recoder.avg)

