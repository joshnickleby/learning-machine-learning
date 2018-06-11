
class List:
    items = []

    def __init__(self, *items):
        self.items = items

    def map(self, fn=lambda item: item):
        self.items = [fn(item) for item in self.items]
        return self
