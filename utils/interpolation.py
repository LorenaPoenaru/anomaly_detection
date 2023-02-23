import pandas as pd

# function that finds the indexes of non-anomalies for interpolation 
def interpolation_indexes(mylist, mynumber):
    
    left_neighbour = 0
    right_neighbour = 0
    
    # check left neighbour
    if((mynumber - 1) not in mylist):
        left_neighbour = mynumber - 1
    else:
        min_number = mynumber
        while min_number in mylist:
            min_number = min_number - 1
        left_neighbour = min_number
    
    # check right neighbour
    if((mynumber + 1) not in mylist):
        right_neighbour = mynumber + 1
    else:
        max_number = mynumber
        while max_number in mylist:
            max_number = max_number + 1
        right_neighbour = max_number
    
    return left_neighbour, right_neighbour


def train_anomaly_removal(df_train):
    
    # extract indexes for anomalies
    indexes = list(df_train[df_train.is_anomaly == 1].index)

    # creating a new df that replaces the anomalous samples with interpolation value
    df = pd.DataFrame(columns = df_train.columns)
    for i in range(0, len(df_train)):

        #print(i)

        # add all non-anomalies
        if(df_train.is_anomaly[i] == 0):
            df = df.append({'timestamp' : df_train.timestamp[i], 'value' : df_train.value[i], 'is_anomaly' : df_train.is_anomaly[i]},
            ignore_index = True)

        if((df_train.is_anomaly[i]==1) & (i != (len(df_train)-1)) & (i != 0)):
            if(df_train.is_anomaly[i+1]!=1):
                
                if((interpolation_indexes(indexes, i)[0] != -1) & (interpolation_indexes(indexes, i)[1] != -1)):
                    value_interpolation = (df_train.value[interpolation_indexes(indexes, i)[0]]
                                           +df_train.value[interpolation_indexes(indexes, i)[1]])/2

                    df = df.append({'timestamp' : df_train.timestamp[i], 'value': value_interpolation, 'is_anomaly' : 0.0}, ignore_index = True)
    return df