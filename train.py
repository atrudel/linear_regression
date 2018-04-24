from LinearRegression import LinearRegression
import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser(description="Select the hyperparameters and \
            the visualization options for training the linear regression model")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                    help="Set the learning rate")
parser.add_argument("-i", "--iter", type=int, default=1500,
                    help="Set the number of iterations")
parser.add_argument("-v", "--visual", action="store_true",
                    help="Plot the training data and the regression line")
parser.add_argument("-c", "--cost", action="store_true",
                    help="Plot the cost function during training")
args = parser.parse_args()

data = pd.read_csv("data.csv")
model  = LinearRegression()
print("Training linear regression model...")
model.train(data['km'], data['price'], lr=args.learning_rate, n_iter=args.iter,
            visual=args.cost)

if args.visual:
    model.display(data['km'], data['price'])

print("Saving model trained model...")
with open("model.pickle", 'wb') as f:
    pickle.dump(model,f)
