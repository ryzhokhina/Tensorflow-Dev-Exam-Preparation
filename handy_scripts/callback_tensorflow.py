MC = tf.keras.callbacks.ModelCheckpoint(
    './content/drive/MyDrive/Colab Notebooks/Sertificate_preparation/Books/Callbacks/Models/mnist_h5.h5',
    monitor='val_loss',
    save_best_only='True',
    verbose=1
)

'''
It will interrupt training when it measures no progress on the validation set for
a number of epochs (defined by the patience argument), and it will optionally roll
back to the best model.
'''
ES = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights='True'
)

'''
At the beginning of every epoch, this callback gets the updated learning rate value from schedule function provided at __init__, 
with the current epoch and current learning rate, and applies the updated learning rate on the optimizer.
'''
LR = tf.keras.callbacks.LearningRateScheduler\
    (lambda epoch: 1e-5 * 10 ** (epoch/2), verbose=1)
'''
he ReduceLROnPlateau callback montiors a specified metric and when that metric stops improving, 
it reduces the learning rate by a specified factor (e.g. divides the learning rate by 10).
'''
RLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.2,
    patience=2,
    verbose=1,
    min_lr=1e-7)