import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train_model import predict_model
from mpl_toolkits.mplot3d import Axes3D


def acceptance_rate_threshold(y_probs, acceptance_rate:int, plot=False):
    """
    Visualize the distribution of the probabilities of default
    Parameter:
    ----------
        - acceptance_rate: int -> the number of loan we want to accept
    Output:
    -------
        Return the threshold according to the acceptance rate
    """
    
    
    # Calculate the threshold according to the right acceptance rate
    threshold = np.quantile(y_probs, acceptance_rate)
    
    if plot == True:
        plt.hist(y_probs, bins=40)
        # Add a reference line to the plot for the threshold
        plt.axvline(x=threshold, color='red', label="threshold")
        plt.title("Pobabilities of default distribution")
        plt.legend()
        plt.show()
    return threshold.round(3)

def bad_rate_calculation(X_test, y_test, y_probs, pred_loans_status, threshold, acceptance_rate):
    
    # Create dataframe
    test_preds_df = pd.DataFrame({
        'true_loan_status': y_test,
        'prob_default': y_probs,
        'pred_loans_status': pred_loans_status,
        'loan_amount': X_test["AMT_CREDIT"],
        'loss_given_default': 1.0, # we consider that the exposure corresponds to the total value of the loan, and that the loss given default is 100% (value 1). This means that a default on each loan represents a loss of the entire amount.
    })
    
    # Apply acceptance rate threshold
    test_preds_df["pred_loans_status"] = test_preds_df["prob_default"].apply(lambda x: 1 if x > threshold else 0)

    # Create a subset of only accepted loans
    accepted_loans = test_preds_df[test_preds_df["pred_loans_status"] == 0]
    
    # Calculate the bad rate
    bad_rate = np.sum(accepted_loans['true_loan_status']) / accepted_loans['true_loan_status'].count()
    
    # Evaluate the impact of the acceptance rate impact
    avg_loan_amount = np.mean(test_preds_df["loan_amount"])
    
    crosstab_result = pd.crosstab(test_preds_df["true_loan_status"], 
                                  test_preds_df["pred_loans_status"])
    
    crosstab_result = crosstab_result.apply(lambda x: x * avg_loan_amount, axis=0)
    
    print("--"*50)
    print("Acceptance rate: ", acceptance_rate.round(3))
    print("Bad Rate:", bad_rate.round(3))
    print("Threshold: ", threshold)
    print("nb accepted loan:", len(accepted_loans))
    print(crosstab_result)
    
    return bad_rate.round(3), test_preds_df, avg_loan_amount.round(2)
    
def plot_portfolio_credit_risk_strategy(strategy_table, elev=30, azim=30):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extracting the columns for plotting
    X = strategy_table["Acceptance Rate"]
    Y = strategy_table["Bad Rate"]
    Z = strategy_table["Estimated Value"]

    # Plotting the surface
    surf = ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')

    # Setting the view angle
    ax.view_init(elev=elev, azim=azim)

    # Adding labels
    ax.set_xlabel('Acceptance Rate')
    ax.set_ylabel('Bad Rate')
    ax.set_zlabel('Estimated Value')
    ax.set_title('3D Surface Plot of Credit Risk Strategy')

    # Adding color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()
    
def plot_strategy_portfolio_optimization(strategy_table):
    # Plot the strategy curve
    plt.plot(strategy_table["Acceptance Rate"], strategy_table["Bad Rate"])
    plt.xlabel("Acceptance Rate")
    plt.ylabel("Bad Rate")
    plt.title("Acceptance and Bad Rate")
    plt.grid(True)
    plt.show()
    
    # Plot the strategy curve
    plt.plot(strategy_table["Acceptance Rate"], strategy_table["Estimated Value"])
    plt.xlabel("Acceptance Rate")
    plt.ylabel("Estimated Value")
    plt.title("Acceptance and Bad Rate")
    plt.grid(True)
    plt.show()
    
    
    # Plot the boxplot
    plt.figure(figsize=(12, 8))  # Increase the size of the boxplot
    strategy_table[["Acceptance Rate", "Threshold", "Bad Rate"]].boxplot()
    plt.show()
    
    
def port_optimization_strategy(model, X_test, y_test):
    acceptance_rates = np.arange(1.0, 0.0, -0.05)
    
    thresholds = list()
    bad_rates = list()
    num_accepted_loans = list()  
    avg_loan_amounts = list()
    
    # Predict probabilities for default
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Predict probabilities for default
    pred_loans_status = model.predict(X_test)
    
    # populate the arrays for the strategy table
    for rate in acceptance_rates:
        #Calculate the threshold for the acceptance rate
        threshold = acceptance_rate_threshold(y_probs, rate, plot=True)
        
        # Add threshold value to the list of thresholds
        thresholds.append(threshold)
        
        # Calculate the bad rate 
        bad_rate, test_preds_df, avg_loan_amount = bad_rate_calculation(X_test, y_test, y_probs, pred_loans_status, threshold, rate)
        bad_rates.append(bad_rate)
        
        # Add the avg_loan_amount
        avg_loan_amounts.append(avg_loan_amount)
        
        # Calculate the number of accepted loans
        accepted_loans_count = len(test_preds_df["prob_default"] < np.quantile(test_preds_df["prob_default"], rate))
        num_accepted_loans.append(accepted_loans_count)
    
    # Create a data frame of the strategy table
    strategy_table = pd.DataFrame(zip(acceptance_rates, bad_rates, thresholds, num_accepted_loans, avg_loan_amounts),
                                  columns=["Acceptance Rate", "Bad Rate", "Threshold", "Num Accepted Loans", "Avg Loan Amount"])
    
    # Estimating portfolio value
    strategy_table["Estimated Value"] = ((strategy_table["Num Accepted Loans"] * (1 - strategy_table["Bad Rate"])) * strategy_table["Avg Loan Amount"]) - (strategy_table["Num Accepted Loans"] * strategy_table["Bad Rate"] * strategy_table["Avg Loan Amount"])
    
    print("Strategy Table")
    print(strategy_table)
    print("\n")
    
    # print the row with the max estimated value
    print("Max estimated value")
    print(strategy_table.loc[strategy_table["Estimated Value"] == np.max(strategy_table["Estimated Value"])])
    print("\n")
    
    # Compute the toal expected loss
    test_preds_df["expected_loss"] = test_preds_df['prob_default'] * test_preds_df["loss_given_default"] * test_preds_df["loan_amount"]
    
    total_expected_loss = round(np.sum(test_preds_df["expected_loss"]), 2)
    
    print("Total expected loss: ", "${:,.2f}".format(total_expected_loss))
    
    return strategy_table