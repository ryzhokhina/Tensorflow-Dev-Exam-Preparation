import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

"""==============================================="""
"""     ImageDataGenerator                        """
"""==============================================="""
def example_using_IDG() :
    '''
    Example using ImageDataGenerator
    :return:
    '''

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    model.fit(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
    return

def load_IDG(path_data):
    '''
    return DirectoryIterator!
    :param path_data:
    :return:
    '''
    """
    load data using ImageDataGenerator
    ImageDataGenerator can load and agmentation data
    It is slower than image_dataset_from directory method
    """
    data_gen = ImageDataGenerator(rescale= 1./255,
                               rotation_range=0.4,
                               horizontal_flip=True,
                               zoom_range = 0.3)
    data = data_gen.flow_from_directory(path_data,target_size=[224,224], batch_size=32, shuffle=True)
    return data



"""==============================================="""
"""     tensorflow_datasets                       """
"""==============================================="""
def load_from_tfds(name):
    '''
    return tf.data.Dataset!
    We can apply a lot function to tf.data.Dataset
    :param name:
    :return:
    '''

    (train, test) = tfds.load(name, split=['train[:10%]', 'test[:10%]'],shuffle_files=True, as_supervised=True)
    print(train.cardinality())
    print(test.cardinality())
    return train, test

def preprocess_image(image, label, img_shape=224):
    image = tf.image.resize(image, [img_shape, img_shape])
    # we can add augmentation there
    return tf.cast(image, tf.float32), label


def preprocess_data(train_data, val_data):
    train_data = train_data \
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(1000) \
        .batch(32) \
        .prefetch(tf.data.AUTOTUNE)

    val_data = val_data \
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(32) \
        .prefetch(tf.data.AUTOTUNE)

    return train_data, val_data

"""==============================================="""
"""     tensorflow_datasets                       """
"""==============================================="""

def load_from_directory(path):
    '''
     return tf.data.Dataset
    :param path:
    :return:
    '''
    data = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=[224, 224], shuffle=True
    )
    return data

if __name__ == '__main__':
    print('start')
    print("load using ImageDataGenerator")
    path_train = '/Users/annaryzhokhina/PycharmProjects/TensorFlowSertificatePreparation/TransferLearning_ImageClasification/data/101_food_classes_10_percent/train'
    data_train = load_IDG(path_train)

    path_test = '/Users/annaryzhokhina/PycharmProjects/TensorFlowSertificatePreparation/TransferLearning_ImageClasification/data/101_food_classes_10_percent/test'
    data_test = load_IDG(path_test)

    print("load mnist from tensorflow datasets")
    load_from_tfds('mnist')

    print("load using image_dataset_from_directory")
    data = load_from_directory(path_train)
    print(data.cardinality())


    print('finish')