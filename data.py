import numpy as np

def get_training_data():
    features = [
    "wears_i_love_javascript_tshirt",
    "is_carrying_menacing_rubber_duck",
    "tries_to_pay_with_pennies",
    "jumps_the_queue",
    ]

    # Each line represents a person, and whether or not they have the features above
    dataset = [
        [1, 1, 1, 1],  
        [1, 1, 1, 0],          
        [1, 1, 0, 0],  
        [0, 1, 1, 0],  
        [0, 0, 0, 1],  
        [0, 1, 0, 0],  
        [0, 0, 0, 0],  
    ]

    # This list represents whether the people above are (0 = not allowed in) or (1 = allowed in)
    targets = [0, 0, 0, 1, 1, 1, 1]  

    
    X = np.array(dataset)
    y = np.array(targets)
    # print(X)
    # print(y)
    return X, y

# def get_training_data():
#     features = [
#     "under_18",
#     "wearing_glasses",
#     "come_in_a_group_of_5",
#     "have_no_shoes_on",
#     "have_no_shirt_on",
#     ]

#     # Each line represents a person, and whether or not they have the features above
#     dataset = [
#         [1, 1, 1, 1, 1],  
#         [1, 1, 1, 0, 0],          
#         [1, 1, 0, 0, 1],  
#         [0, 1, 0, 1, 0],  
#         [1, 0, 0, 0, 1],
#         [0, 0, 0, 0, 0],
#         [0, 1, 1, 0, 0],
#         [1, 1, 0, 1, 1],
#         [0, 1, 1, 1, 0],
#         [0, 1, 0, 0, 1],
#         [1, 0, 0, 0, 0],
#         [0, 1, 1, 1, 1],
#     ]

#     # This list represents whether the people above are (0 = not allowed in) or (1 = allowed in)
#     targets = [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0]  

    
#     X_train = np.array(dataset)
#     y_train = np.array(targets)
#     # print(X)
#     # print(y)
#     return X_train, y_train

