from LinearRegression import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Select additional options")
    parser.add_argument("-r", "--reset", action="store_true",
                        help="Erase previously trained model")
    parser.add_argument("-v", "--visual", action="store_true",
                        help="Show prediction in a graph among the dataset")
    return parser.parse_args()

def plot_estimation(mileage, price):
    data = pd.read_csv("data.csv")
    fig = plt.figure("Data visualization")
    fig.suptitle("Data visualization and one model estimation")
    ax = plt.axes()
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price ($)")
    plt.scatter(data['km'], data['price'])
    plt.scatter(mileage, price)
    plt.show()


args = parse_args()
# Welcome message
print("Welcome.")
print("This program can estimate the price of a car according to its mileage.")

# Erase previously trained model if --reset option was selected
if args.reset:
    os.remove("model.pickle")
    print("Deleting previous model...")

# Recover previously trained model, or create a new one
try:
    with open("model.pickle", 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = LinearRegression()
    print("(The model isn't trained...)")

# Prompt user input and process it
try:
    user_input = input("\nPlease enter a mileage (km): ")
    mileage = float(user_input)
    if mileage >= 0:
        price = model.predict(mileage)
        print("Estimated price: $%d" %(price))
        if args.visual:
            plot_estimation(mileage, price)
    else:
        print("Mileage must be a positive number.")
except ValueError:
    print("Mileage must be a number.")
