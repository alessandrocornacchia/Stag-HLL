import simpy 

def process(env):
    while True:
        yield env.timeout(1)
        print('elapsed')

def iterations(env):
    for i in range(10):
        yield env.timeout(1)
        print(i)


class A:
    a = 5

    def change(self):
        A.a = 4

print(A.a)
A().change()
print(A.a)
env = simpy.Environment()
env.process(process(env))
it = env.process(iterations(env))
env.run(until=it)