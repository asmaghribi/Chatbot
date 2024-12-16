import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
from train import vectorizer, label_encoder, model, train_vectors, train_labels, X_train, y_train, X_test, y_test

unique_labels = np.unique(label_encoder.inverse_transform(train_labels))
num_classes = len(unique_labels)
print("Nombre de classes:", num_classes)
print(unique_labels)

Y_pred_train = model.predict(X_train)
# Obtention de la matrice de confusion de training_set
cm_train = confusion_matrix(y_train, Y_pred_train)

print("Accuracy(train)={:.3f}".format(accuracy_score(y_train,Y_pred_train)))
print(cm_train)

# Définition des xticks labels
plt.xticks(ticks=np.arange(len(unique_labels)), labels=unique_labels)

# Affichage de la matrice de confusion pour le training_set
plt.imshow(cm_train, cmap='Blues')
plt.colorbar()
plt.show()

# Matrice de confusion pour le test_set
Y_pred_test = model.predict(X_test)
cm_test = confusion_matrix(y_test, Y_pred_test)

print("Accuracy(test)={:.3f}".format(accuracy_score(y_test,Y_pred_test)))

sensitivities = recall_score(y_test, Y_pred_test, average=None)
precisions = precision_score(y_test, Y_pred_test, average=None)

print(cm_test)

# Définition des xticks labels
plt.xticks(ticks=np.arange(len(unique_labels)), labels=unique_labels)

# Affichage de la matrice de confusion pour le test_set
plt.imshow(cm_test, cmap='Blues')
plt.colorbar()
plt.show()
