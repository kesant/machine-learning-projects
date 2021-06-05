import numpy as np
import matplotlib.pyplot as plt

def estimate_b0_b1(x,y):

  np.size(x)
  #calcular el promedio de X e Y
  m_x,m_y=np.mean(x),np.mean(y)
  #calcular las sumatorias de XY y de XX
  sumatoria_xy=np.sum((x-m_x)*(y-m_y))
  sumatoria_xx=np.sum(x*(x-m_x))
  #calculamos la constante b_0 y la pendiente b_1
  b_1=sumatoria_xy/sumatoria_xx
  b_0=m_y-(b_1*m_x)
  #vamos a retornar b_0,b_1 como tupla
  return (b_0,b_1)

#funcion graficado
def funcion_de_graficado(x,y,b):
  #esto nos va graficar una grafica de puntos de color verde con marcadores de color naranja 
  plt.scatter(x,y,color="g",marker="o",s=30)
  #ecuacion de regresion lineal
  y_pred=b[0]+b[1]*x
  #se le da la orden de graficar
  plt.plot(x,y_pred,color="b")
  #Etiquetas
  plt.xlabel("x-Independiente")
  plt.ylabel("y-dependiente")
  #instruccion para graficar la grafica la grafica
  plt.show()

#funcion main()
def main():
  #Data set creamos los arrays de x e y 
  x=np.array([1,2,3,4,5])
  x=np.array([2,3,5,6,5])
  #obtenemos b1 y b0
  b=estimate_b0_b1(x,y)
  print(f"Los valores b0={b[0]} y b1={b[1]}")
  #graficamos la regresion
  funcion_de_graficado(x,y,b)
if __name__=="__main__":
  main()    