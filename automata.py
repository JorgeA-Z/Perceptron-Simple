import numpy as np

class Automata:
    def __init__(self, ruta):
        self.ruta = ruta

    def data(self):
        train = []
        label = []

        archivo = open(self.ruta, mode="r")
        lineas = archivo.readlines()
        for linea in lineas:
                cadenas = linea[:len(linea)-1].split(',')

                x1 = float(cadenas[0])
                x2 = float(cadenas[1])
                x3 =  int(cadenas[2])

                train.append([x1, x2])
                
                label.append(x3)
                

        return np.array(train), np.array(label)




