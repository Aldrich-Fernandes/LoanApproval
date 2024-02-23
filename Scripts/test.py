class P:
    def __init__(self, name):
        self.name = name

    def intro(self):
        print(self.name)

class C1(P):
    def __init__(self, name):
        super().__init__(name)
        self.intro()

p = P("Steve")
c1 = C1("Jack")