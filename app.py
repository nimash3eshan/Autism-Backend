from flask import Flask, request, send_file, jsonify
import librosa
import numpy as np
import tensorflow as tf
import os
import tempfile
import audioread
import soundfile as sf
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Create a folder for saving audio files if it doesn't exist
if not os.path.exists("static/audio"):
    os.makedirs("static/audio")

# Load the model
voice_separation_model = tf.keras.models.load_model(
    "./models/voice_separation_model", compile=False
)

age_model = joblib.load("./models/age_model.pkl")

anomaly_model = joblib.load("./models/anomaly_detector_model.pkl")


def get_magnitude_spectrogram(y, sr, n_fft=1024, hop_length=512):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(D)
    return magnitude, phase


def pad_spectrogram(spec, max_length):
    return librosa.util.pad_center(spec, size=max_length, axis=1)


def spectrogram_to_audio(magnitude, phase, hop_length=512):
    return librosa.istft(magnitude * phase, hop_length=hop_length)


def load_with_audioread(file):
    fd, tmp_filename = tempfile.mkstemp(suffix=".m4a")
    file.save(tmp_filename)

    with audioread.audio_open(tmp_filename) as src:
        sr_native = src.samplerate
        y, _ = librosa.load(tmp_filename, sr=sr_native)

    os.close(fd)  # Close the file descriptor
    try:
        os.remove(tmp_filename)
    except PermissionError:
        print(f"Warning: Could not delete temp file {tmp_filename}")

    return y, sr_native


def extract_mfcc_from_spectrogram(spec, sr=22050, hop_length=512):
    """Convert spectrogram back to audio and extract MFCC."""
    audio_reconstructed = librosa.istft(spec, hop_length=hop_length)
    mfccs = librosa.feature.mfcc(y=audio_reconstructed, sr=sr, n_mfcc=13)
    return mfccs.mean(axis=1)


@app.route("/process-voice", methods=["POST"])
def process_voice():
    print("Processing voice...")
    # Ensure the request has a file attached
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Get the audio file from the request
    audio_file = request.files["file"]

    y, sr = load_with_audioread(audio_file)
    magnitude, phase = get_magnitude_spectrogram(y, sr)

    # Convert the conversation data to the format expected by the model
    X_test = np.transpose(np.array([magnitude]), (0, 2, 1))
    predicted = voice_separation_model.predict(X_test)[0]

    # Ensure the phase information and predicted magnitude are correctly padded to match the expected shape
    max_length = 10031  # this is the expected size

    # Create an array of zeros with the desired shape
    padded_predicted = np.zeros((513, max_length))
    # Fill in the values from the predicted matrix
    predicted_length = min(predicted.shape[1], max_length)
    padded_predicted[:, :predicted_length] = predicted.T[:, :predicted_length]

    phase_matrix = np.ones((513, max_length)) * phase[:, -1].reshape(-1, 1)
    phase_matrix[:, : phase.shape[1]] = phase

    audio = spectrogram_to_audio(padded_predicted, phase_matrix)

    # Save the audio to a temporary file
    _, tmp_filename = tempfile.mkstemp(suffix=".wav")
    sf.write(tmp_filename, audio, sr)

    # Return the audio file
    return send_file(
        tmp_filename,
        as_attachment=True,
        download_name="separated_voice.wav",
        mimetype="audio/wav",
    )


@app.route("/predict-age", methods=["POST"])
def predict_age():
    try:
        # Ensure the request has a file attached
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        # Get the audio file from the request
        audio_file = request.files["file"]

        # Load the audio data
        y, sr = load_with_audioread(audio_file)
        magnitude, _ = get_magnitude_spectrogram(y, sr)

        # Extract MFCCs
        test_mfcc = extract_mfcc_from_spectrogram(magnitude)

        # Predict the age
        age_prediction = age_model.predict([test_mfcc])
        print(age_prediction)  # Print the raw predictions
        predicted_age = round(age_prediction[0])  # Round to the nearest integer
        print(predicted_age)

        return jsonify({"predicted_age": int(predicted_age)})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": "An error occurred during processing."}), 500


@app.route("/detect-anomaly", methods=["POST"])
def detect_anomaly():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        audio_file = request.files["file"]
        y, sr = load_with_audioread(audio_file)
        magnitude, _ = get_magnitude_spectrogram(y, sr)
        test_mfcc = extract_mfcc_from_spectrogram(magnitude)

        # Predict anomalies using the loaded anomaly model
        anomaly_predictions = anomaly_model.predict(np.array([test_mfcc]))

        anomalies = sum([1 for pred in anomaly_predictions if pred == -1])
        percentage_anomalies = (anomalies / len(anomaly_predictions)) * 100

        anomaly_classification = [
            "potential autism" if pred == -1 else "typical"
            for pred in anomaly_predictions
        ]

        response_data = {
            "number_of_detected_anomalies": anomalies,
            "percentage_of_detected_anomalies": percentage_anomalies,
            "anomaly_classification": anomaly_classification,
        }

        return jsonify(response_data)

    except Exception as e:
        print("Error during anomaly detection:", str(e))
        return jsonify({"error": "An error occurred during anomaly detection."}), 500


if __name__ == "__main__":
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit
    app.run(debug=True)
