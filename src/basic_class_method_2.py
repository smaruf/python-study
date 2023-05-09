class Fruit:
    name = 'Fruitas'

    def printName(cls):
            print('The name is:', cls.name)

Fruit.printAge = classmethod(Fruit.printName)
Fruit.printAge()
