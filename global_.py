class Runtime():
    __env = None

    @classmethod
    def get(cls):
        return cls.__env
    
    @classmethod
    def set(cls,value):
        cls.__env = value