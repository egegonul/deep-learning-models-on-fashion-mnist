import numpy as np
import tensorflow as tf
import pickle
from models import *
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('src',type=str)
parser.add_argument('model',type=str)
parser.add_argument('dest',type=str)
args=parser.parse_args()

if __name__=='__main__':
    #load dataset as numpy arrays
    path=args.src
    test_images=np.load(path+"test_images.npy")
    test_labels=np.load(path+"test_labels.npy")
    train_images=np.load(path+"train_images.npy")
    train_labels=np.load(path+"train_labels.npy")

    #dataset rescaled to -1,1
    train_images=train_images / 255.0
    test_images=test_images/255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    model = Models[args.model]

    #this section reshapes the input for the cnn models only
    if(args.model=="cnn_3" or args.model=="cnn_3_2" or args.model=="cnn_5"):
        train_images=train_images.reshape(30000,28,28,1)
        test_images=test_images.reshape(5000,28,28,1)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))



    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  #model compiled with the required settings
    history = model.fit(train_images, train_labels, epochs=15, shuffle=True, validation_split=0.1, ) #fit method is used for training, shuffling and validation split is done within
    accuracy = history.history['accuracy']      #accuracy and loss metrics are obtained from the history object
    loss = history.history['loss']
    val_acc=history.history['val_accuracy']                 #this section runs the training for 1 epoch before the 9 times loop
    top_acc = model.evaluate(test_images, test_labels)[1]
    temp_weights= model.layers[0].get_weights()[0]
    top_weights=temp_weights

    num_iters=1
    for i in range (num_iters):         #15 epochs are run 9 more times, model is reset each time
        model = Models[args.model]
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history=model.fit(train_images, train_labels, epochs=15,shuffle=True,validation_split=0.1,)
        accuracy = [a+b for a,b in zip(accuracy , history.history['accuracy'])]   #metric curves are formed
        loss = [a + b for a,b in zip(loss ,  history.history['loss'])]
        val_acc = [a+b for a,b in zip(val_acc , history.history['val_accuracy'])]
        temp_weights= model.layers[0].get_weights()[0]
        test_acc= model.evaluate(test_images, test_labels)[1]
        if(test_acc>top_acc):      #weights of the best test accuracy model is chosen
            top_weights=temp_weights

    avg_loss=np.divide(loss,num_iters)    #average of the metric curves are taken
    avg_acc=np.divide(accuracy,num_iters)
    avg_val_acc=np.divide(val_acc,num_iters)

    results= {                         #results dictionary formed
        "name":"cnn_3",
        "loss_curve":avg_loss,
        "train_acc_curve":avg_acc,
        "val_acc_curve":avg_val_acc,
        "test_acc":top_acc,
        "weights":top_weights
    }

    print(results)

    dict_file=open(args.dest,"wb")
    pickle.dump(results,dict_file)
    dict_file.close()
