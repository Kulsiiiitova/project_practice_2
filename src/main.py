import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    with open('data/Keylogger_Detection.csv') as file:
        df = pd.read_csv('data/Keylogger_Detection.csv')
        X = df.drop('Class', axis=1)
        Y = df['Class']
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, Y, test_size=0.3, random_state=42, stratify=Y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.666, random_state=42, stratify=y_temp)



        print(f"Обучающая выборка: {X_train.shape[0]} samples")
        print(f"Валидационная выборка: {X_val.shape[0]} samples")
        print(f"Тестовая выборка: {X_test.shape[0]} samples")

if __name__ == '__main__':
    main()
