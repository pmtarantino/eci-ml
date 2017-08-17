import argparse
from training_module import train_set, eval_set

# USO
# python run.py clasificador
# Ejemplo: python run.py rf2

# Clasificadores:

# rf2 : Random forest con max leaf nodes = 2
# rf5 : Random forest con max leaf nodes = 5
# rf5oob : Random forest con max leaf nodes = 5 y oob_score=True

# svc = Support Vector
# nusvc = Nu-Support Vector
# polysvc = Support Vector, Polinomial kernel
# sigsvc = Support Vector, Sigmoid Kernel

parser = argparse.ArgumentParser(description='Clasificador.')
parser.add_argument('clasificador', metavar='C', type=str,
                    help='Tipo de clasificador a utilizar')

args = parser.parse_args()
clf_type = args.clasificador

orig_auc, clf = train_set(clf_type)
loc = eval_set(clf)

random_acum = 0
random_iterations = 50

for i in range(0,random_iterations):
	auc, clf = train_set(clf_type,shuffle=True)
	random_loc = eval_set(clf)

	if random_loc > loc:
		random_acum = random_acum+1

pvalor = 1-(random_acum/random_iterations)

print  '|' , clf_type , '|' , orig_auc, '|' , loc, '|' , pvalor,  '|' 