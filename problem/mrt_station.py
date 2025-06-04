class MrtStation:
    def __init__(self,
                 latitude: float,
                 longitude: float,
                 line: str,
                 address: str,
                 position: str):
        self.latitude = latitude
        self.longitude = longitude
        self.line = line
        self.address = address
        self.position = position
        
    def __repr__(self):
        return f"{self.line} {self.position} {self.address}"
