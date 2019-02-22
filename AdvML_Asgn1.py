
# coding: utf-8

# In[30]:


import pandas as pd
from matplotlib import pyplot as plt 
import tensorflow as tf
import numpy as np
import os
import cv2
import PIL
from PIL import Image


# In[31]:


numberOfImages = 8220
input_array = np.zeros((numberOfImages,227,227), dtype=np.float32)
j = 0
for file in os.listdir("/home/ee/mtech/eet182559/Slices"):
    img = cv2.imread("/home/ee/mtech/eet182559/Slices/"+file,cv2.IMREAD_GRAYSCALE)
    area = (5, 5, 232, 232) 
    img1 = Image.fromarray(img)
    img2 = img1.crop(area)
    normalized_X = np.zeros((227, 227))
    normalized_X = cv2.normalize(np.array(img2), normalized_X , 0, 1, cv2.NORM_MINMAX)
    input_array[j,:,:] = normalized_X
    j += 1


# In[32]:


y = np.genfromtxt('/home/ee/mtech/eet182559/Labels2.csv',delimiter=',')
y = np.asarray(y)
y = y.reshape(8220,2)


# In[33]:


count = 1403
dataAugmentation_input = np.zeros((count*4,227,227), dtype=np.float32)
dataAugmentation_output = np.zeros((count*4,2), dtype=np.float32)
j = 0
k = 0
for i in y:
    if(i[0] == 0):
        dataAugmentation_output[j][0] = 0
        dataAugmentation_output[j][1] = 1
        img = Image.fromarray(input_array[k,:,:])
        rot = img.rotate(45)
        dataAugmentation_input[j] = np.array(rot)
        j += 1
        dataAugmentation_output[j][0] = 0
        dataAugmentation_output[j][1] = 1
        rot = img.rotate(90)
        dataAugmentation_input[j] = np.array(rot)
        j += 1
        dataAugmentation_output[j][0] = 0
        dataAugmentation_output[j][1] = 1
        rot = img.rotate(135)
        dataAugmentation_input[j] = np.array(rot)
        j += 1
        dataAugmentation_output[j][0] = 0
        dataAugmentation_output[j][1] = 1
        flip = np.fliplr(input_array[k,:,:])
        dataAugmentation_input[j] = flip
        j += 1
    k += 1


# In[34]:


newInput = np.vstack((input_array, dataAugmentation_input))
newOutput = np.vstack((y, dataAugmentation_output))


# In[35]:


newInput.shape


# In[36]:


count*4 + 8220


# In[37]:


totalImages = count*4 + numberOfImages
input = newInput.reshape(totalImages,227*227);
print(input.shape)
out = np.hstack((input,newOutput))
np.random.shuffle(out)


# In[38]:


split = 12500
splity = len(out[0, : ]) - 2
train_X,train_y = out[ : split , : splity], out[ : split, splity : ]
train_X = train_X.reshape(split,227*227);
train_Y = train_y.reshape(split,2)
print(train_X.shape)
print(train_Y.shape)

validation_X, validation_Y = out[split:,:splity], out[split : , splity : ]
validation_X = validation_X.reshape(totalImages - split, 227*227);
validation_Y = validation_Y.reshape(totalImages - split, 2)
print(validation_X.shape)
print(validation_Y.shape)


# In[39]:


# Placeholder variable for the input images
x = tf.placeholder(tf.float32, shape=[None, 227*227], name='X')
# Reshape it into [num_images, img_height, img_width, num_channels]
x_image = tf.reshape(x, [-1, 227, 227, 1])  

# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


# In[40]:


def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
    
    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # Add the biases to the results of the convolution.
        layer += biases
        
        return layer, weights


# In[41]:


def new_pool_layer(input, name):
    
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        return layer


# In[42]:


def new_relu_layer(input, name):
    
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)
        
        return layer


# In[43]:


def new_fc_layer(input, num_inputs, num_outputs, name):
    
    with tf.variable_scope(name) as scope:

        # Create new weights and biases.
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        
        # Multiply the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases
        
        return layer


# In[44]:


# Convolutional Layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=1, filter_size=5, num_filters=6, name ="conv1")

# Pooling Layer 1
layer_pool1 = new_pool_layer(layer_conv1, name="pool1")

# RelU layer 1
layer_relu1 = new_relu_layer(layer_pool1, name="relu1")

batch_norm1 = tf.layers.batch_normalization(layer_relu1, training=True, momentum=0.9)

# Convolutional Layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=batch_norm1, num_input_channels=6, filter_size=5, num_filters=16, name= "conv2")

# Pooling Layer 2
layer_pool2 = new_pool_layer(layer_conv2, name="pool2")

# RelU layer 2
layer_relu2 = new_relu_layer(layer_pool2, name="relu2")

batch_norm2 = tf.layers.batch_normalization(layer_relu2, training=True, momentum=0.9)

# Flatten Layer
num_features = batch_norm2.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_relu2, [-1, num_features])

# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=50, name="fc1")

#dropout
dropped = tf.layers.dropout(layer_fc1,0.5)

# RelU layer 3
layer_relu3 = new_relu_layer(dropped, name="relu3")

batch_norm3 = tf.layers.batch_normalization(layer_relu3, training=True, momentum=0.9)


# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=batch_norm3, num_inputs=50, num_outputs=2, name="fc2")


# In[45]:


# Use Softmax function to normalize the output
with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, dimension=1)


# In[46]:


# Use Cross entropy cost function
with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)


# In[47]:


# Use Adam Optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


# In[48]:


# Accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[49]:


# Initialize the FileWriter
writer = tf.summary.FileWriter("Training_FileWriter/")
writer1 = tf.summary.FileWriter("Validation_FileWriter/")


# In[50]:


# Add the cost and accuracy to summary
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()


# In[51]:


num_epochs = 40
batch_size = 200
import time


# In[25]:


with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    
    # Loop over number of epochs
    for epoch in range(num_epochs):
        start_time = time.time()

        current_batch = 0
        train_accuracy = 0
        
        for batch in range(0, int(len(train_X)/batch_size)):
            
            # Get a batch of images and labels
            x_batch = train_X[current_batch:current_batch+batch_size, :]
            y_true_batch = train_Y[current_batch:current_batch+batch_size]
            
            # Put the batch into a dict with the proper names for placeholder variables
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            
            # Run the optimizer using this batch of training data.
            sess.run(optimizer, feed_dict=feed_dict_train)
            
            # Calculate the accuracy on the batch of training data
            train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
            
            current_batch = current_batch + batch_size
        
          
        train_accuracy /= int(len(train_X)/batch_size)
        
        # Generate summary and validate the model on the entire validation set
        vali_accuracy = sess.run(accuracy, feed_dict={x:validation_X, y_true:validation_Y})
        #writer1.add_summary(summ, epoch)
        
        end_time = time.time()

        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")

        print("\tAccuracy:")
        print ("\t- Training Accuracy:\t{}".format(train_accuracy))
        print ("\t- Validation Accuracy:\t{}".format(vali_accuracy))

    saver = tf.train.Saver()
    saver.save(sess,'/home/ee/mtech/eet182559/SessionTrain4',global_step=1, write_meta_graph=True)
    


# In[24]:



# Test Input
count = 0;
numberOfImages = 942
test_input = np.zeros((numberOfImages,227,227), dtype=np.float32)
test_labels = np.zeros((numberOfImages,2), dtype=np.float32)
j=0
for file in os.listdir("/home/ee/mtech/eet182559/normalsVsAbnormalsV1/abnormalsJPG"):
    img = cv2.imread("/home/ee/mtech/eet182559/normalsVsAbnormalsV1/abnormalsJPG/" + file, cv2.IMREAD_GRAYSCALE)
    test_input[j,:,:] = img
    test_labels[j,0] = 1. 
    test_labels[j,1] = 0. 
    j += 1

for file in os.listdir("/home/ee/mtech/eet182559/normalsVsAbnormalsV1/normalsJPG"):
    img = cv2.imread("/home/ee/mtech/eet182559/normalsVsAbnormalsV1/normalsJPG/" + file, cv2.IMREAD_GRAYSCALE)
    test_input[j,:,:] = img
    test_labels[j,0] = 0. 
    test_labels[j,1] = 1. 
    j += 1


# In[26]:



with tf.Session() as sess:
    saver = tf.train.import_meta_graph('/home/ee/mtech/eet182559/SessionTrain4-1.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    
    current_batch = 0
    train_accuracy = 0
    x_batch = test_input.reshape(942, 227*227)
    y_true_batch = test_labels
            
    feed_dict_train = {x: x_batch, y_true: y_true_batch}

    testAccuracy = sess.run(accuracy, feed_dict=feed_dict_train)


# In[27]:


print("Test Accuracy: " + str(testAccuracy))

