import keras

model = keras.models.load_model("final_model.keras")

# Most .keras models store this
print(model.get_config().get("_keras_version"))