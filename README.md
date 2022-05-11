# Audio_classification

from google.colab import drive
import librosa
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np

#Nota: Verificar que existe la ruta: /content/drive/MyDrive/UrbanSound8K
drive.mount('/content/drive')


# cargar el modelo en la siguiente dirección:
modelo = load_model('/content/saved_models/audio_classification.hdf5')

#cargar la dirección del archivo de audio a predecir
filename="/content/drive/MyDrive/Uninorte/Todo_cuenta_uninorte/Puntos1-50/Splits/P5splits/Punto 01 P5.wav_segment103.wav"

#cargamos el audio y extraemos coeficientes mfcc
audio, sample_rate = librosa.load(filename, res_type='kaiser_best') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)

predictions = modelo.predict(mfccs_scaled_features)  # Realiza las prediccion a los coeficientes del archivo de audio seleccionado
clase_predicha = predictions.argmax(axis=-1)  # Obtiene las clase predicha a partir de la función argmax
print(clase_predicha)

#predicción label
#prediction_class = labelencoder.inverse_transform(clase_predicha) 
#prediction_class
