from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

# Compute the class weights
class_weights = compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)

# Convert the class weights to a dictionary
class_weight_dict = dict(enumerate(class_weights))

# Use the class weights in the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    class_weight=class_weight_dict
)