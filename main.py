import numpy as np
import perceptron
import automata
import matplotlib.pyplot as plt

if __name__ == "__main__":

    a = automata.Automata("XOR_trn.csv");
    b = automata.Automata("XOR_tst.csv");

    # Datos de entrenamiento y prueba (XOR)
    training_data, training_labels = a.data()
    test_data, test_labels = b.data()


    print("Training data and training labels")
    print(training_data, training_labels)
    print("Test data and Test labels")
    print(test_data, test_labels)

    # Crea un perceptrón con 2 entradas (para el operador XOR)
    p = perceptron.perceptron(num_inputs=2, learning_rate=0.1, epochs=100)

    # Entrena el perceptrón
    p.train(training_data, training_labels)

    # Prueba el perceptrón
    correct_predictions = 0
    total_predictions = len(test_data)

    predicted_labels = []

    for inputs, label in zip(test_data, test_labels):
        prediction = p.predict(inputs)
        predicted_labels.append(prediction)
        print(f"Entradas: {inputs}, Predicción: {prediction}")
        if prediction == label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")


    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5

    # Crear una malla de puntos para la visualización
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    mesh_data = np.c_[xx.ravel(), yy.ravel()]

    # Calcular las predicciones del perceptrón en la malla de puntos
    Z = np.array([p.predict(point) for point in mesh_data])
    Z = Z.reshape(xx.shape)

    # Visualizar los datos de prueba
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.6)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, marker='o', s=25)
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.title("Agrupación de datos y Frontera de Decisión en Escala de Grises")
    plt.show()

    