import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model("model.h5")

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    "data/train",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(test_data.classes, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_data.class_indices.keys()))
disp.plot(xticks_rotation=45)

plt.savefig("confusion_matrix.png")