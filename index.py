from services.preprocessing import PreProcessing
from services.split import Split
from src.models.logistic_regression import run as logistic_regression
from src.models.knn import run as knn
from src.models.random_forest import run as random_forest
from src.models.svm import run as svm
from src.models.xg_boost import run as xg_boost

data  = PreProcessing().process()
folds = Split(data).get_folds()

# logistic_regression(data, folds)
# knn(data, folds)
random_forest(data, folds)
# svm(data, folds)
# xg_boost(data, folds)