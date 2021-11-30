import os
import Data_Extracting
import NeuralNetwork


def main():
    print("In main function")
    #Data_Extracting.Get_HN_Patients()
    NeuralNetwork.Train(os.path.join(os.getcwd(), "Training_Data"))


if __name__ == "__main__":
    main()    