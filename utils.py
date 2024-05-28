import numpy as np

# Функція для генерації бінарної матриці з однією одиницею на вказаному індексі
def generate_binary_matrix(index, length=25):
    """
    Args:
        index (int): Індекс, який потрібно встановити в 1.
        length (int, optional): Довжина матриці. За замовчуванням 25.
    
    Returns:
        np.ndarray: Бінарна матриця з однією одиницею на індексі.
    """
    # Створюємо нульову матрицю заданої довжини
    binary_matrix = np.zeros(length, dtype=float)
    # Встановлюємо елемент на позиції index в 1
    binary_matrix[index] = 1
    return binary_matrix

# Функція для отримання індексу з бінарної матриці з однією одиницею
def retrieve_index_from_binary_matrix(binary_matrix):
    """
    Args:
        binary_matrix (np.ndarray): Бінарна матриці з однією одиницею.
    
    Returns:
        int: Індекс.
    """
    # Знаходимо індекс елемента, що дорівнює 1
    index = np.where(binary_matrix == 1)[0]
    # Повертаємо перший індекс, якщо він існує, інакше повертаємо None
    return index[0] if len(index) > 0 else None

if __name__ == '__main__':
    # Приклад використання
    index = 6
    # Генеруємо матрицю з однією одиницею на індексі
    generated_binary_matrix = generate_binary_matrix(index)
    print("Сгенерована бінарна матриця:")
    print(generated_binary_matrix)

    # Отримуємо індекс з сгенерованої бінарної матриці
    retrieved_index = retrieve_index_from_binary_matrix(generated_binary_matrix)
    print("Отриманий індекс:")
    print(retrieved_index)
