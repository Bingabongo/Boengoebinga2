# Import
from preprocessData import preprocessData
from computeFeatures import computeFeatures
from util import scorer
from util import printScores
from util import createSplits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import csv

"""
This script contains the pipeline for predicting the presence of bad smell.

The dataset that we will use is from the following URL:
- https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction

For the background of this project, please check the following paper:
- https://arxiv.org/pdf/1912.11936.pdf

This script mainly uses the scikit-learn package, documented in the URL below:
- https://scikit-learn.org/stable/

Our task is to train a model (denoted F) to predict presence of bad smell.
We will use the term "smell event" to indicate the "presence of bad smell".
The model (F) maps a set of features (X) to a response (Y), where Y=F(X).
The features are extracted from the raw data from air quality and weather sensors.
The response means if there will be a bad smell event in the future.

How can we define a bad smell event?
We define it as if the sum of smell ratings within a time range exceeds a threshold.
This reflects if many people reports bad smell ratings within a future time range.
Details will be explained as you move forward to read this script.

The following is a brief description of the pipeline:
- Step 1: Preprocess the raw data
- Step 2: Select variables from the preprocessed sensor data
- Step 3: Extract features (X) and the response (Y) from the preprocessed data
- Step 4: Train and evaluate a machine learning model (F) that maps X to Y
- Step 5: Investigate the importance of each feature
"""


# This is a reusable function to print the data
def pretty_print(df, message):
    print("\n================================================")
    print("%s\n" % message)
    print(df)
    print("\nColumn names below:")
    print(list(df.columns))
    print("================================================\n")


"""
Step 1: Preprocess the raw data

We want to preprocess sensor and smell data and get the intermediate results.
This step does the following:
- Load the sensor and smell data in the dataset folder
- Merge the sensor data from different air quality and weather monitoring stations
- Align the timestamps in the sensor and smell data by resampling the data points
- Treat missing data

The returned variable "df_sensor" means the preprocessed sensor data.
The "DateTime" column means the timestamp
, which has the format "year-month-day hour:minute:second+timezone".
All the timestamps in "df_sensor" is in the GMT timezone.
Other columns mean the average value of the sensor data in the previous hour.
For example, "2016-10-31 06:00:00+00:00" means October 31 in 2016 at 6AM UTC time.
Column "3.feed_1.SO2_PPM" means the averaged SO2 values from 5AM to 6AM.
The column name suffix SO2 means sulfur dioxide, and PPM means the unit.
The prefix "3.feed_1." in the column name means a specific sensor (feed ID 1).
You can ignore the "3." at the begining of the column name.
Here is what it means for each feed ID:
- Feed 26: Lawrenceville ACHD
- Feed 28: Liberty ACHD
- Feed 23: Flag Plaza ACHD
- Feed 43: Parkway East ACHD
- Feed 11067: Parkway East Near Road ACHD
- Feed 1: Avalon ACHD
- Feed 27: Lawrenceville 2 ACHD
- Feed 29: Liberty 2 ACHD
- Feed 3: North Braddock ACHD
- Feed 3506: BAPC 301 39TH STREET BLDG AirNow
- Feed 5975: Parkway East AirNow
- Feed 3508: South Allegheny High School AirNow
- Feed 24: Glassport High Street ACHD

You can get the metadata and location of the feed by searching the feed ID below:
- https://environmentaldata.org/

Some column names look like "3.feed_11067.SIGTHETA_DEG..3.feed_43.SIGTHETA_DEG".
This means that the column has data from two sensor stations (feed ID 11067 and 43).
The reason is that some sensor stations are replaced by the new ones over time.
So in this case, we merge sensor readings from both feed ID 11067 and 43.
Here is a list of the explanation of column name suffix:
- SO2_PPM: sulfur dioxide in ppm (parts per million)
- SO2_PPB: sulfur dioxide in ppb (parts per billion)
- H2S_PPM: hydrogen sulfide in ppm
- SIGTHETA_DEG: standard deviation of the wind direction
- SONICWD_DEG: wind direction (the direction from which it originates) in degrees
- SONICWS_MPH: wind speed in mph (miles per hour)
- CO_PPM: carbon monoxide in ppm
- CO_PPB: carbon monoxide in ppb
- PM10_UG_M3: particulate matter (PM10) in micrograms per cubic meter
- PM10B_UG_M3: same as PM10_UG_M3
- PM25_UG_M3: fine particulate matter (PM2.5) in micrograms per cubic meter
- PM25T_UG_M3: same as PM25_UG_M3
- PM2_5: same as PM25_UG_M3
- PM25B_UG_M3: same as PM25_UG_M3
- NO_PPB: nitric oxide in ppb
- NO2_PPB: nitrogen dioxide in ppb
- NOX_PPB: sum of of NO and NO2 in ppb 
- NOY_PPB: sum of all oxidized atmospheric odd-nitrogen species in ppb
- OZONE_PPM: ozone (or trioxygen) in ppm
- OZONE: same as OZONE_PPM

More explanation about the suffix is in the following URL:
- https://tools.wprdc.org/pages/air-quality-docs.html

The returned variable "df_smell" means the preprocessed smell data.
Similar to "df_sensor", "df_smell" also has the "DateTime" column indicating timestamps.
The timestamps also have the same format as in the "df_sensor" variable.
Other columns mean the sum of smell ratings within an hour in a specific zipcode.
For example, the "15217" column indicates the zipcode 15217 in Pittsburgh, Pennsylvania.
In the latest row, the timestamp is "2018-09-30 05:00:00+00:00"
, which means this row contains the data from 4:00 to 5:00 on September 30 in 2018.
For example, on this row, column "15217" has value 5
, which means there is a smell report with rating 5 in the above mentioned time range.
Notice that the data ignored all smell ratings from 1 to 2.
This is becasue we only want the ratings that indicate "bad" smell.
For more description about the smell, please check the following URL:
- https://smellpgh.org/how_it_works

Both "df_sensor" and "df_smell" use the pandas.DataFrame data structure.
More information about the data structure is in the following URL:
- https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe
"""

# Preprocess and print sensor and smell data
# (no need to modify this part)
df_sensor_raw, df_smell = preprocessData(
    in_p=["dataset/esdr_raw/", "dataset/smell_raw.csv"])
pretty_print(df_sensor_raw, "Display all sensor data and column names")
pretty_print(df_smell, "Display smell data and column names")

# Create feature sets
PM = [
    '3.feed_1.PM25B_UG_M3..3.feed_1.PM25T_UG_M3',
    '3.feed_3.PM10B_UG_M3',
    '3.feed_23.PM10_UG_M3',
    '3.feed_24.PM10_UG_M3',
    '3.feed_26.PM25B_UG_M3',
    '3.feed_26.PM10B_UG_M3',
    '3.feed_29.PM10_UG_M3',
    '3.feed_29.PM25_UG_M3',
    '3.feed_3506.PM2_5',
    '3.feed_3508.PM2_5',
    '3.feed_5975.PM2_5',
    '3.feed_11067.PM25T_UG_M3..3.feed_43.PM25T_UG_M3',
]

H2S = [
    '3.feed_1.H2S_PPM',
    '3.feed_28.H2S_PPM',
]

wind = [
    '3.feed_1.SIGTHETA_DEG',
    '3.feed_1.SONICWD_DEG',
    '3.feed_1.SONICWS_MPH',
    '3.feed_26.SONICWS_MPH',
    '3.feed_26.SONICWD_DEG',
    '3.feed_26.SIGTHETA_DEG',
    '3.feed_28.SIGTHETA_DEG',
    '3.feed_28.SONICWD_DEG',
    '3.feed_28.SONICWS_MPH',
    '3.feed_3.SONICWD_DEG',
    '3.feed_3.SONICWS_MPH',
    '3.feed_3.SIGTHETA_DEG',
    '3.feed_11067.SIGTHETA_DEG..3.feed_43.SIGTHETA_DEG',
    '3.feed_11067.SONICWD_DEG..3.feed_43.SONICWD_DEG',
    '3.feed_11067.SONICWS_MPH..3.feed_43.SONICWS_MPH',
]

SO2 = [
    '3.feed_1.SO2_PPM',
    '3.feed_27.SO2_PPB',
    '3.feed_28.SO2_PPM',
    '3.feed_3.SO2_PPM',
]

CO = [
    '3.feed_23.CO_PPM',
    '3.feed_27.CO_PPB',
    '3.feed_11067.CO_PPB..3.feed_43.CO_PPB',
]

ozone = [
    '3.feed_26.OZONE_PPM',
    '3.feed_3506.OZONE',
]

NO_X_Y = [
    '3.feed_27.NO_PPB',
    '3.feed_27.NOY_PPB',
    '3.feed_11067.NO2_PPB..3.feed_43.NO2_PPB',
    '3.feed_11067.NOX_PPB..3.feed_43.NOX_PPB',
    '3.feed_11067.NO_PPB..3.feed_43.NO_PPB',
]

# Preset parameters
pred_hour = 21
threshold_hour = 1
look_back_hr = 5

# Feature importance. Choose which features to use
good_features = PM + wind + SO2 + CO + ozone + NO_X_Y
print(f'{len(good_features)} features total')
print(good_features)

# Iterate over all features that are chosen
for feature in good_features:
    feature_set = ['DateTime'] + good_features

    # Remove 1 feature from the feature set and train and test
    feature_set.remove(feature)
    print(f'removed feature {feature}')
    df_sensor = df_sensor_raw[feature_set]

    # Plot and save correlation matrix of variables and results.
    # Uncomment if want to be used
    # correlation(df_esdr=df_sensor, df_smell=df_smell)

    # Plot smell ratings over time
    # Uncomment if want to be used
    # smell_ratings(df_smell=df_smell)

    # Plot sensor values over time
    # Uncomment if want to be used
    # sensor_plots(df_sensor=df_sensor)

    # Print the selected sensor data
    # (no need to modify this part)
    # pretty_print(df_sensor, "Display selected sensor data and column names")

    # Indicate the number of future hours to predict smell events
    # (you may want to modify this parameter for experiments)
    smell_predict_hrs = pred_hour  # range 1-12 increments of 1 . best = 24

    # Indicate the number of hours to look back to check previous sensor data
    # (you may want to modify this parameter for experiments)
    look_back_hrs = look_back_hr  # range 2-36 increments of 1. best = 19

    # Indicate the threshold to define a smell event
    # (you may want to modify this parameter for experiments)
    smell_thr_per_hour = threshold_hour  # range 3-30 increments of 3. best = 1
    smell_thr = smell_predict_hrs * smell_thr_per_hour

    # Indicate if you want to add interaction terms in the features (like x1*x2)
    # (you may want to modify this parameter for experiments)
    add_inter = False

    # Compute and print features (X) and response (Y)
    # (no need to modify this part)
    df_X, df_Y, _ = computeFeatures(df_esdr=df_sensor,
                                    df_smell=df_smell,
                                    f_hr=smell_predict_hrs,
                                    b_hr=look_back_hrs,
                                    thr=smell_thr,
                                    add_inter=add_inter)
    # pretty_print(df_X, "Display features (X) and column names")
    # pretty_print(df_Y, "Display response (Y) and column names")

    # Indicate how much data you want to use to test the model
    # (you may want to modify this parameter for experiments)
    # test_size = 168
    # test_size = 744*3 # 3 months

    test_ratio = 3.5  # range 0.25-4.5 increments of .25

    test_size = int(744 * test_ratio)  # 1 months

    # Indicate how much data you want to use to train the model
    # (you may want to modify this parameter for experiments)
    # train_size = 8760 # 1 year
    train_ratio = 4 * test_ratio
    train_size = int(744 * train_ratio)  # 4 months
    # train_size = 336

    # Build the cross validation splits
    # (no need to modify this part)
    # print(f'shape: {df_X.shape[0]}')
    splits = createSplits(test_size, train_size, df_X.shape[0])

    # Indicate which model you want to use to predict smell events
    # (you may want to modify this part to use other models)

    # model = DummyClassifier(strategy="constant", constant=0)
    # model = DecisionTreeClassifier()  # 2sec
    model = RandomForestClassifier()  # 20sec

    # Perform cross-validation to evaluate the model
    # (no need to modify this part)
    # for model in [DecisionTreeClassifier(), RandomForestClassifier()]:
    # print("Use model", model)
    # print("Perform cross-validation, please wait...")
    result = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=scorer)
    printScores(result)

    # Calculate test scores
    accuracy = sum(result['test_accuracy']) / len(result['test_accuracy'])
    precision = sum(result['test_precision']) / len(result['test_precision'])
    recall = sum(result['test_recall']) / len(result['test_recall'])
    f1 = sum(result['test_f1']) / len(result['test_f1'])  # gives the wrong f1-score
    f1_2 = (2 * (precision * recall)) / (precision + recall)
    print(f'f1 {f1_2}')

    # Save model and scores to csv file
    with open('tests_4.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            model, feature, smell_thr_per_hour, smell_predict_hrs, look_back_hrs, train_size,
            test_size, accuracy, precision, recall, f1, f1_2
        ])
