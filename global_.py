class Runtime():
    __env = None
    __eos = None

    @classmethod
    def get(cls):
        return cls.__env
    
    @classmethod
    def set(cls,value):
        cls.__env = value
        cls.__eos = value.event()

    @classmethod
    def end_sim(cls):
        return cls.__eos

    @classmethod
    def terminate(cls):
        cls.__eos.succeed()