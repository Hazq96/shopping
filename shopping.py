import csv
import sys


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Bcolser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    import calendar
    evidence = []
    labels = []

    # You can check this later.
    month_to_number = {name: num - 1 for num, name in enumerate(calendar.month_abbr) if num}

    with open(filename) as data:
        reader = csv.reader(data)
        # Skip the first row.
        next(reader)

        for col in reader:
            evidence.append([
                int(col[0]),
                float(col[1]),
                int(col[2]),
                float(col[3]),
                int(col[4]),
                float(col[5]),
                float(col[6]),
                float(col[7]),
                float(col[8]),
                float(col[9]),
                month_to_number[col[10][:3]],
                int(col[11]),
                int(col[12]),
                int(col[13]),
                int(col[14]),
                1 if col[15] == 'Returning_Visitor' else 0,
                int((col[16]) == 'TRUE'),
            ])

            labels.append(
                int((col[17]) == 'TRUE')
            )

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity, specificity = 0.0, 0.0
    positive, negative = 0.0, 0.0

    sensitivity, specificity = 0.0, 0.0
    positive, negative = 0.0, 0.0

    for label, prediction in zip(labels, predictions):
        if label == 1:  # if label is positive, then we calculate sensitivity (positive rate)
            positive += 1.0
            if label == prediction:
                sensitivity += 1.0

        else:  # if label is negative, then we calculate specificity (negative rate)
            negative += 1.0
            if label == prediction:
                specificity += 1.0

    sensitivity /= positive
    specificity /= negative

    return sensitivity, specificity


if __name__ == "__main__":
    main()
