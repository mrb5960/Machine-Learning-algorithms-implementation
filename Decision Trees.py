import csv
import numpy as np

# initializing global variables that will be used in recursion
level = -1  # number of recursion levels (depth)
tabs = 0    # tabs to be left before if-else statements
out_file = None # file handler used for writing files

def find_best_threshold(dataset):
    '''
    method that finds best threshold by calculating minimum entropy for each element in all attributes
    :param dataset: the dataset containing the attributes
    :return: best_threshold: the threshold with the minimum weighted entropy
             attribute: the attribute from which the best_threshold was chosen
    '''

    # initializing variables
    min_weighted_entropy = 999999
    best_threshold = 99999
    attribute = 99999

    # loop to go through each attribute
    for col in range(0, 4):

        # loop to go through each element of the attribute
        for outer_row in range(0, len(dataset)):

            # set the value of threshold as that of the current element
            thresh = dataset[outer_row][col]

            # variables that will be used to calculate entropy and weighted entropy
            lt_zeros = 0
            lt_ones = 0
            ge_zeros = 0
            ge_ones = 0
            lt_record_count = 0
            ge_record_count = 0

            # loop to compare threshold with each element of the attribute
            for row in range(0, len(dataset)):

                # comparing value to threshold and checking the target variable to increment count
                # this will be used to calculate entropy and weighted entropy
                if dataset[row][col] <= thresh and dataset[row][4] == 0:
                    lt_record_count += 1
                    lt_zeros += 1
                elif dataset[row][col] > thresh and dataset[row][4] == 0:
                    ge_record_count += 1
                    ge_zeros += 1
                elif dataset[row][col] <= thresh and dataset[row][4] == 1:
                    lt_record_count += 1
                    lt_ones += 1
                elif dataset[row][col] > thresh and dataset[row][4] == 1:
                    ge_record_count += 1
                    ge_ones += 1

            # to avoid division by zero error
            if lt_zeros == 0 or lt_ones == 0:
                lt_entropy = 0
            else:
                lt_entropy = - ((lt_zeros/(lt_zeros + lt_ones) * np.log2(lt_zeros/(lt_zeros + lt_ones))) + (lt_ones/(lt_zeros + lt_ones) * np.log2(lt_ones/(lt_zeros + lt_ones))))
            lt_weight = lt_record_count/len(dataset)

            if ge_zeros == 0 or ge_ones == 0:
                ge_entropy = 0
            else:
                ge_entropy = - ((ge_zeros/(ge_zeros + ge_ones) * np.log2(ge_zeros/(ge_zeros + ge_ones))) + (ge_ones/(ge_zeros + ge_ones) * np.log2(ge_ones/(ge_zeros + ge_ones))))
            ge_weight = ge_record_count/len(dataset)

            # calculating the weighted entropy
            weighted_entropy = (lt_weight * lt_entropy) + (ge_weight * ge_entropy)

            #print('Row: ',outer_row,'Element: ', dataset[outer_row][col],'we: ', round(weighted_entropy,3),'lt_zeros: ', lt_zeros, 'lt_ones: ', lt_ones, 'ge_zeros: ', ge_zeros, 'ge_ones: ', ge_ones, 'lt_record_count: ', lt_record_count, 'ge_record_count: ', ge_record_count)

            # keeping track of minimum weighted entropy, the element and attribute the element belongs to
            if weighted_entropy <= min_weighted_entropy:
                min_weighted_entropy = weighted_entropy
                best_threshold = thresh
                attribute = col

    return best_threshold, attribute

def split(dataset, threshold, col):
    '''
    method which is used to split the dataset based on the given threshold value
    :param dataset: the dataset containing the attributes
    :param threshold: the threshold based on which the data is to be split
    :param col: the column number of the attribute to which the threshold belongs
    :return: left_split: list of rows where attribute value is less than threshold
             right_split: list of rows where attribute value is more than threshold
    '''

    # initializing two 2D lists, left and right
    left_split = []
    right_split = []

    # splitting the entire dataset based on the threshold value
    for row in range(0, len(dataset)):

        # if the value of the element in col is less than threshold add entire row to left list
        if dataset[row][col] <= threshold:
            left_split.append(dataset[row])

        # else add it to the right list
        else:
            right_split.append(dataset[row])

    return left_split, right_split

def decision_tree_builder(dataset):
    '''
    method that forms the if-else statements(decisions)
    :param dataset: the dataset containing the attributes
    :return: None
    '''

    # accessing the global variables
    global level, tabs, out_file
    level += 1
    tabs += 1

    # finding the best threshold for splitting the dataset
    best_threshold, attribute = find_best_threshold(dataset)

    # splitting the dataset into 2 parts using the threshold value
    left_split, right_split = split(dataset, best_threshold, attribute)

    # lists that contains just the target values(class) which will be used for checking the purity
    left_target = [row[4] for row in left_split]
    right_target = [row[4] for row in right_split]

    # writing the rules to the output program file
    out_file.write('\n')
    # inserting tabs depending on the recursion level
    out_file.write('\t'*tabs)
    # writing the if statement and appending the attribute number and threshold value to the rule
    out_file.write('if row[{}] <= {}:'.format(attribute, best_threshold))

    # for left sub tree
    # if class is pure stop recursion
    if all(num == left_target[0] for num in left_target):
        #print('Level: ', level, ', Attribute: ', attribute+1, ', Threshold <= ', best_threshold, ', Class: ', left_target[0])
        out_file.write('\n')
        out_file.write('\t'*(tabs+1))
        # write return statement along with the class value to the output program file
        out_file.write('return \'{}\''.format(int(left_target[0])))

    # if class is not pure, continue recursion
    else:
        decision_tree_builder(left_split)

    # adding newline
    out_file.write('\n')
    # adding tabs
    out_file.write('\t'*tabs)
    # adding else condition
    out_file.write('else:')

    # for right sub tree
    # if class is pure stop recursion
    if all(num == right_target[0] for num in right_target):
        #print('Level: ', level, ', Attribute: ', attribute+1, ', Threshold > ', best_threshold, ', Class: ', right_target[0])
        out_file.write('\n')
        out_file.write('\t'*(tabs+1))
        # write return statement along with the class value to the output program file
        out_file.write('return \'{}\''.format(int(right_target[0])))

    # if class is not pure, continue recursion
    else:
        decision_tree_builder(right_split)

    # decrementing the values of the variables after returning from recursion level
    tabs -= 1
    level -= 1


def emit_prologue(output_file_handle):
    '''
    method that writes the headers to the output program file
    :param output_file_handle: file writer object used to write to the file
    :return: None
    '''
    output_file_handle.write('import csv')


def emit_body(output_file_handle, training_data):
    '''
    method that writes the body of the program to the output program file
    The body contains other method definitions, such as classifier() and main()
    It also makes a call to the decision_tree_builder() method which writes the
    actual decisions to the output program file
    :param output_file_handle: file writer object used to write to the file
    :param training_data: dataset that is used to train the classifier
    :return: None
    '''

    # defining classifier()
    output_file_handle.write('\n\ndef classifier(row):')

    # making a call to the method that trains the classifier and generates rules and writes it
    # to the output program file
    decision_tree_builder(training_data)

    # defining main()
    output_file_handle.write('\n\ndef main():')
    output_file_handle.write('\n\tpath = \'C:/abc.csv\'')

    # opening the csv file that contains the validation data
    output_file_handle.write('\n\ttesting_data = list(csv.reader(open(path)))')

    # converting all values to float
    output_file_handle.write('\n\n\tfor row in range(1,len(testing_data)):')
    output_file_handle.write('\n\t\tfor col in range(0, len(testing_data[1])):')
    output_file_handle.write('\n\t\t\ttesting_data[row][col] = float(testing_data[row][col])')

    # removing headers
    output_file_handle.write('\n\ttesting_data.pop(0)')

    # opening the csv file that will contain the classifications
    output_file_handle.write('\n\n\twith open(\'C:/abc.csv\', \'w\', newline=\'\') as output:')
    output_file_handle.write('\n\t\twriter = csv.writer(output)')
    output_file_handle.write('\n\n\t\tfor row in testing_data:')

    # making a call to the classifier() which contains the decision rules for each row of validation data
    output_file_handle.write('\n\t\t\ttarget = classifier(row)')

    # printing the result to the console
    output_file_handle.write('\n\t\t\tprint(target)')

    # writing the result to the csv file
    output_file_handle.write('\n\t\t\twriter.writerow(target)')


def emit_epilogue(output_file_handle):
    '''
    method that writes code to make a call to the main()
    :param output_file_handle: file writer object used to write to the file
    :return: None
    '''
    output_file_handle.write('\n\nmain()')

def main():
    '''
    method that accesses the training data and makes calls to emit_prologue(), emit_body() and emit_epilogue()
    :return: None
    '''

    # accessing the global variable
    global out_file

    # path of the trainer dataset
    path = 'C:/xyz.csv'

    # path of the output program file
    out_path = 'C:/abc.py'

    # opening the output program file in write mode
    out_file = open(out_path, 'w')

    # get 2D array from csv file
    training_data = list(csv.reader(open(path)))

    # casting from string to float
    for row in range(1,len(training_data)):
        for col in range(0, len(training_data[1])):
            training_data[row][col] = float(training_data[row][col])

    # removing headers
    training_data.pop(0)

    emit_prologue(out_file)
    emit_body(out_file, training_data)
    emit_epilogue(out_file)

main()
