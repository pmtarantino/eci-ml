# Identificar estados de proposal basados en un set de entrenamiento de sueño.

## Instalación

Clonar el repositorio y correr

`pip install -r reqs.txt`

para instalar todas las dependencias de Python (mejor si se hace dentro de un virtualenv para evitar conflictos de versionado con otras librerías ya instaladas en el sistema)

## Ejecutar

Para ejecutar, simplemente correr

`python run.py clasificador`

donde los clasificadores disponibles son:

De Random forest:
- rf2 : Random forest con max leaf nodes = 2
- rf5 : Random forest con max leaf nodes = 5
- rf5oob : Random forest con max leaf nodes = 5 y oob_score=True

Support vector machine:
- svc = Support Vector Machine clasico, con un kernel de base radial (default)
- nusvc = Nu-Support Vector Machine, similar a svc pero con una cota de 0.1 en los errores de entrenamiento
- polysvc = Support Vector, con kernel polinomial
- sigsvc = Support Vector, con kernel Sigmoid 

El polysvc usa un polinomio de grado 3 por default, para utilizar con otros grados de 2 a 6), se utiliza polysvc2, polysvc4, polysvc5 o polysvc6.

## Resultados

Para todos los clasificadores seleccionados, se entrenó con el set de gente durmiendo y despierta, y luego se lo analizó contra la gente a la que se le administró proposol.

Los resultados son los siguientes:

| Clasificador  | AUC entrenamiento | AUC Proposol | P-Valor |
| ------------- | ------------- | ------------- | ------------- |
| rf2 | 0.972222222222 | 0.722222222222 | 1 |
| rf5 | 0.966475095785 | 0.708333333333 | 1 |
| rf5oob | 0.970785440613 | 0.652777777778 | 1 |
| svc | 0.966475095785 | 0.861111111111 | 1 |
| nusvc | 0.997126436782 | 0.833333333333 | 1 |
| polysvc | 0.892720306513 | 0.888888888889 | 1 |
| sigsvc | 0.974137931034 | 0.861111111111 | 1 |
| polysvc5 | 0.919540229885 | 0.777777777778 | 1 |
| polysvc2 | 0.955938697318 | 0.861111111111 | 1 |
| polysvc1 | 0.974137931034 | 0.861111111111 | 1 |
| polysvc4 | 0.941570881226 | 0.916666666667 | 1 |
| polysvc6 | 0.853448275862 | 0.777777777778 | 1 |

(Para calcular el p-valor, se hicieron 50 iteraciones con los datos re-ordenados azarosamente y se contó la cantidad de veces que el área bajo la curva era mayor que con los datos sin reordenar. En todos los casos, esto valor siempre fue menor).

Como puede verse, los valores de AUC de los clasificadores de Random Forest oscilan alrededor del valor 0.7.

Ya para los clasificadores de support vector machine se obtienen resultados más interesantes. Con el clasificador default, nu-svc y con kernel sigmoid, se tienen valores medios de 0.85.

Lo interesante es con el support vector machine con kernel polinomial y cómo se va alterando de acuerdo a los grados del polinomio:

| Grados | AUC |
| ------------- | ------------- |
|1|0.861111111111|
|2|0.861111111111|
|3|0.888888888889|
|4|0.916666666667|
|5|0.777777777778|
|6|0.777777777778|

Se nota un pico en el polinomio de grado 4, y como para ambos lados decrece.

De esta forma, con un área bajo la curva de más 0.9, pareciera que el clasificador de support vector machine con un kernel polinomial de grado 4 es la mejor alternativa para este caso particular.
