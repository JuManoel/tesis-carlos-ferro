from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# funcion auxiliar para acalcular las metricas de los modelos 
# de classificacion binaria
def calculate_metrics(y_true, y_pred):

    aux = [1 if i > 0 else 0 for i in y_pred] # devido a  BCEWithLogitsLoss no aplico Sigmoid en el modelo
    accuracy = accuracy_score(y_true, aux)
    precision = precision_score(y_true, aux)
    recall = recall_score(y_true, aux)
    f1 = f1_score(y_true, aux)
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4)
    }