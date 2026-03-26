from data import load_data
from model import preprocess, create_sequences, build_lstm

def train_model():
    data = load_data()
    scaled, scaler = preprocess(data)

    X, y = create_sequences(scaled)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_lstm()
    model.fit(X, y, epochs=5, batch_size=32)

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    return data, predictions

if __name__ == "__main__":
    train_model()
