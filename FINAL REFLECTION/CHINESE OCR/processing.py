import numpy
import glob # os function for accessing all files inside a directory

def byte_decoder(directory):
    x, y = [], [] # initialization of data storage: x = character bitmaps, y = response feature (int/char)
    characters = [] # character array used for indexing responses

    for filename in glob.iglob(f'{directory}/*'): # iterates over all files in {directory}
        
        cx, cy = [], [] # current bitmap/character set, reset for each file
        with open(filename, 'rb') as data: # with open() processes and closes each file (by byte: rb)
            print(filename) # status check for processing, as it takes around 10 minutes for all of the data

            bytes_listed = [] # all bytes per file
            offset = 0 # position in bytes array
            while (byte := data.read(1)): 
                bytes_listed.append(byte) # breaking file data into array

            while offset < len(bytes_listed): # iterating over each byte, while loop is preferred here due to the need to process multiple bytes simultaneously  
                current_sample_size = bytes_listed[offset] + bytes_listed[offset + 1] + bytes_listed[offset + 2] + bytes_listed[offset + 3] # unused data point keeping track of distance to next item in dataset, can be used to replace missing data
                offset += 4

                current_character = 0 # reset char value
                current_character, characters = character_sort(byte_to_num((bytes_listed[offset] + bytes_listed[offset + 1])), characters) 
                cy.append(current_character) # adds corrected character to current storage
                offset += 2

                current_bitmap_size = 0 # initializes bitmap parameters
                current_bitmap_height = 0
                current_bitmap_width = 0 
                current_bitmap_height = byte_to_num((bytes_listed[offset] + bytes_listed[offset + 1]))
                current_bitmap_width = byte_to_num((bytes_listed[offset + 2] + bytes_listed[offset + 3]))
                current_bitmap_size = current_bitmap_width * current_bitmap_height
                offset += 4

                current_bitmap = [] # initializes bitmap
                current_bitmap_size_tracker = 0
                while current_bitmap_size_tracker < current_bitmap_size: # iterates over every byte in bitmap
                    current_bitmap.append(byte_to_num(bytes_listed[offset + current_bitmap_size_tracker]))
                    current_bitmap_size_tracker += 1
                cx.append(dimensionality_buffer(current_bitmap, current_bitmap_width, current_bitmap_height)) # converts list -> numpy array and pads/crops values
                offset += current_bitmap_size
                current_bitmap_size = 0
        for item in cx:
            x.append(item) # combining current bitmap with total array
        for item in cy:
            y.append(item) # combining current response values with total array 

def character_sort(character, y): # function for matching each character/symbol to an integer value
    for i in range(0, len(y)): # for each character already indexed
        if character == y[i]: # if the current character is == that index, return the index of the character
            return i, y
    y.append(character) # or add the character as a new value to the array
    return len(y) - 1, y # len(y) - 1 returns the correct character index (-1 makes len() start at 0 instead of 1), y is the store character array

def byte_to_num(byte):
    return int.from_bytes(byte, byteorder='little', signed=False) # converts from byte(hex escape) to int

def dimensionality_buffer(byte_list, width, height):
    byte_array = numpy.array(byte_list).reshape((width, height)) # converts byte list to 2 dimensional array
    shape = numpy.shape(byte_array) 
    padded_array = numpy.full((500,500), 255) # creates padding array of size (500,500) with blank values 255
    padded_array[:shape[0],:shape[1]] = byte_array # superimposes bitmap (bytearray) data onto blank padding set
    return padded_array[0:100, 0:100] # crops final output from (500,500) to (100,100) based on average dimension values

def collect_data():
    trainX, trainY = byte_decoder( f'C:/Users/edarling23/Downloads/casia/Gnt1.0Train') # calls decoding function for each directory: training data
    testX, testY = byte_decoder(f'C:/Users/edarling23/Downloads/casia/Gnt1.0Test') # testing data/dir

    return trainX, trainY, testX, testY # returns data for model use

collect_data() # function used for testing pre-procssing above, activates when file is imported so is removed for training :)

##### code snippets below: used for testing ! #####

 #for i in range(171):
     #   del trainX[0]
      #  del trainY[0]
       # del testX[0]
        #del testY[0]
    
#byte_test = b'\xb6\xf3'
#print(byte_test.decode('gb18030'))

#pain = b'\xb0\xa1'
#print(pain.decode('gbk'))

# text1 = open('/Users/evelyndarling/python/DATASETS/HANDWRITTEN/Gnt1.0TrainPart1/001-f.gnt', 'rb')
# print(text1)
# notes lol

# byte_decoder('/Users/evelyndarling/python/DATASETS/HANDWRITTEN/Gnt1.0TrainPart1/001-f.gnt')
#print(b'\x21\x00'.decode('utf-8'))
#def byte_decoder(address):
 #   x, y = [], []
  #  bytes_listed = []
  #  data = open(address, 'rb')
  #  while (byte := data.read(1)):
  #      bytes_listed.append(byte)
  #      testcounter += 1
  ##      num = int.from_bytes(byte, byteorder='little', signed=False)
  #      print(num)
  #      print(byte)
  #      if testcounter == 6:
  #          break
  #  return x, y

   #for bingus in trainY:
     #   print(bingus)
      #  bingus = bingus.decode('gbk')
       # counter += 1
      #  print(bingus)
      #  print(counter)
      #  if counter == 10:
      #      break

              #print(trainY[i].decode('gbk'))
    
    #for char in trainY:
     #   char = char.decode('gbk')
    #for char in testY:
     #   char = char.decode('gbk')

    #matplotlib.pyplot.scatter(widths, heights)
    #matplotlib.pyplot.show()
    #print(trainX[0])
    #print(trainY[0])
    #matplotlib.pyplot.imshow(trainX[0])
    #matplotlib.pyplot.show()

#    print(trainX.count)

 #   print(trainY.count)
  #  test_array = numpy.array(trainX)
   # test_array2 = numpy.array(trainY)
   # print(test_array.shape)
   # print(test_array2.shape)

# CASIA-HWDB

# USER EXPERIENCE AND ETHICAL ANALYSIS

# In comparison with many of the subsets of machine learning, Optical Character Recognition has little 
# impact onto the wellbeing of its users. It does not determine ones societal worth, diagnose life-
# threatening disease, or threaten labor work. Still, there are a few key areas where its impact can be 
# negative -> depending on the accuracy, training, and application of the model and its decisions.

# To describe the potential use cases and harm that could be caused as a result of them, I will outline 
# my main idea for the application of my model.
# -------------------------------------------------------------------------------- #
  # I would like to use OCR to develop a language learning application. The app 
  # (built in SwiftUI) would integrate with TensorFLow using CoreML. The app would 
  # allow users to practice Chinese by drawing characters, at which point the app 
  # abstracts the drawing into a bitmap and returns the most likely character that 
  # has been drawn
# -------------------------------------------------------------------------------- #

# Communication is essential, and any application that influences ones ability to communicate can have 
# negative effects. If the model has too low an accuracy, then it has the potential to mislead its users 
# and incorrectly identify characters. If the knowledge that the user gained from the application were to 
# be used in a serious setting, such as a court case, it could cause life changing error. It is nearly 
# impossible to create a classification algorithm with 100% accuracy. The bulk of the ethical dilemma 
# occurs in consideration of the line that must be drawn between accuracy and inaccuracy. With a large 
# enough pool of users, detrimental error is inevitable. Is 98% accuracy sufficient? 99%? 99.5%? The most 
# ethical route of action would likely be to tell the user that the model's results should not be trusted; 
# however, at that point, there isn't really a point in using the model at all. As an auxiliary point, the 
# issue could be diminished by making the model return a prediction of the most likely values, as opposed 
# to a single character - increasing its chance of guessing the correct value (provided that the model has 
# a low enough loss, of course)

# Additionally, there is the issue of representation within the training data. Chinese is heavily influenced 
# by dialect, and as such, there are variations in characters across regions. CASIA-HWDB, the set used for 
# this analysis, was compiled by 320 - 420 writers (depending on the percentage of data accessed) from Beijing. 
# The data does not fully encompass the specific styles and patterns of more rural regions.