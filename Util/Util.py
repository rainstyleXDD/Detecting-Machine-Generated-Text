import numpy as np

'''
    parameters: X - features, y - labels , k - k number of Train/Validation set
    output: a list of k number of Train/Validation set
    
'''
def get_k_val(X, y, k):
    output = []
    
    samples_per_class_in_test = 250  # Adjust as needed

    # Initialize empty lists to store the train and test indices
    train_indices1 = [[] for i in range(k)]
    
    test_indices1 = [[] for i in range(k)]
    
    
    for class_label in [0,1]:
        # Get the indices of samples belonging to the current class
        class_indices1 = np.where(d1y == class_label)[0]
        

        # Randomly select samples_per_class_in_test samples from this class
        selected_indices1 = np.random.choice(class_indices1, samples_per_class_in_test*k, replace=False)
        
        
        selected_indices1 = selected_indices1.reshape(k, samples_per_class_in_test)
        
        
        for i in range(k):
            test_indices1[i].extend(selected_indices1[i])
           
            

            # Add the remaining samples to the train set indices
            remaining_indices = np.setdiff1d(class_indices1, selected_indices1[i])
            train_indices1[i].extend(remaining_indices)
            
            
    for i in range(k):
        # Split the data into train and test sets using the selected indices
        output.append([X[train_indices1[i]], X[test_indices1[i]], y[train_indices1[i]], y[test_indices1[i]]])
    
    return output


'''
    parameters: d1x - features for domain 1, d1y - labels for domain 1, d2x - features for domain 2, d2y - labels for domain 2, k - k number of Train/Validation set
    output: a list of k number of Train/Validation set
    
'''
def get_k_val_2_domain(d1x, d1y, d2x, d2y, k):
    output = []
    
    samples_per_class_in_test = 250  # Adjust as needed

    # Initialize empty lists to store the train and test indices
    train_indices1 = [[] for i in range(k)]
    train_indices2 = [[] for i in range(k)]
    test_indices1 = [[] for i in range(k)]
    test_indices2 = [[] for i in range(k)]
    
    for class_label in [0,1]:
        # Get the indices of samples belonging to the current class
        class_indices1 = np.where(d1y == class_label)[0]
        class_indices2 = np.where(d2y == class_label)[0]

        # Randomly select samples_per_class_in_test samples from this class
        selected_indices1 = np.random.choice(class_indices1, samples_per_class_in_test*k, replace=False)
        selected_indices2 = np.random.choice(class_indices2, samples_per_class_in_test*k, replace=False)
        
        selected_indices1 = selected_indices1.reshape(k, samples_per_class_in_test)
        selected_indices2 = selected_indices2.reshape(k, samples_per_class_in_test)
        
        for i in range(k):
            test_indices1[i].extend(selected_indices1[i])
            test_indices2[i].extend(selected_indices2[i])
            

            # Add the remaining samples to the train set indices
            remaining_indices = np.setdiff1d(class_indices1, selected_indices1[i])
            train_indices1[i].extend(remaining_indices)
            
            remaining_indices = np.setdiff1d(class_indices2, selected_indices2[i])
            train_indices2[i].extend(remaining_indices)
    for i in range(k):
        # Split the data into train and test sets using the selected indices
        output.append([d1x[train_indices1[i]].append(d2x[train_indices2[i]], ignore_index = True), d1x[test_indices1[i]].append(d2x[test_indices2[i]], ignore_index= True), d1y[train_indices1[i]].append(d2y[train_indices2[i]], ignore_index= True), d1y[test_indices1[i]].append(d2y[test_indices2[i]], ignore_index= True)])
    
    return output
