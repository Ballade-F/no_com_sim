class RingBuffer:
    def __init__(self, max_length):
        self.max_length = max_length
        self.queue = []
        self.next = 0

    def push(self, item):
        if len(self.queue) < self.max_length:
            self.queue.append(item)
            self.next = (self.next +1) % self.max_length
        else:
            self.queue[self.next] = item
            self.next = (self.next + 1) % self.max_length

    # 0 表示最新的元素， -1表示最老的元素， 1表示第二新的元素
    def __getitem__(self, index):
        # 判断是否为空
        if not self.queue:
            raise IndexError("RingBuffer is empty")
        if index >= len(self.queue) or index < -len(self.queue):
            raise IndexError("index out of RingBuffer range")
        adjusted_index = (self.next - index - 1) % len(self.queue)
        return self.queue[adjusted_index]

    def __len__(self):
        return len(self.queue)


if __name__ == "__main__":
    # Example usage
    pq = RingBuffer(4)
    pq.push(1)
    pq.push(2)
    pq.push(3)
    print("expect 3,3,2,1,1")
    print(len(pq)) # 3
    print(pq[0]) # 3
    print(pq[1]) # 2
    print(pq[2]) # 1
    print(pq[-1]) # 1

    pq.push(4)
    print("expect 4,4,3,2,1")
    print(len(pq)) # 4
    print(pq[0]) # 4
    print(pq[1]) # 3
    print(pq[2]) # 2
    print(pq[-1]) # 1

    pq.push(5)
    print("expect 4,5,4,3,2")
    print(len(pq)) # 4
    print(pq[0]) # 5
    print(pq[1]) # 4
    print(pq[2]) # 3
    print(pq[-1]) # 2

    pq.push(6)
    pq.push(7)
    pq.push(8)
    pq.push(9)
    pq.push(10)
    print("expect 4,10,9,8,7")
    print(len(pq)) # 4
    print(pq[0]) # 10
    print(pq[1]) # 9
    print(pq[2]) # 8
    print(pq[-1]) # 7

