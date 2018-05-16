'''
Examining The SparkContext

In this exercise you'll get familiar with the SparkContext.

You'll probably notice that code takes longer to run than you might expect. This is because Spark is some serious software. It takes more time to start up than you might be used to. You may also find that running simpler computations might take longer than expected. That's because all the optimizations that Spark has under its hood are designed for complicated operations with big data sets. That means that for simple or small problems Spark may actually perform worse than some other solutions!
Instructions
100xp

Get to know the SparkContext.

    Call print() on sc to verify there's a SparkContext in your environment.
    print() sc.version to see what version of Spark is running on your cluster.

    Take Hint (-30xp)
'''
# Verify SparkContext
print(____)

# Print Spark version
print(____)

'''
Creating a SparkSession

We've already created a SparkSession for you called spark, but what if you're not sure there already is one? Creating multiple SparkSessions and SparkContexts can cause issues, so it's best practice to use the SparkSession.builder.getOrCreate() method. This returns an existing SparkSession if there's already one in the environment, or creates a new one if necessary!
Instructions
100xp

    Import SparkSession from pyspark.sql.
    Make a new SparkSession called my_spark using SparkSession.builder.getOrCreate().
    Print my_spark to the console to verify it's a SparkSession.

    Take Hint (-30xp)
'''
# Import SparkSession from pyspark.sql
from ____ import ____

# Create my_spark
my_spark = ____

# Print my_spark
print(____)


'''
Viewing tables

Once you've created a SparkSession, you can start poking around to see what data is in your cluster!

Your SparkSession has an attribute called catalog which lists all the data inside the cluster. This attribute has a few methods for extracting different pieces of information.

One of the most useful is the .listTables() method, which returns the names of all the tables in your cluster as a list.
Instructions
100xp

    See what tables are in your cluster by calling spark.catalog.listTables() and printing the result!

    Take Hint (-30xp)
'''
# Print the tables in the catalog
print(spark.____.____())


'''
Are you query-ious?

One of the advantages of the DataFrame interface is that you can run SQL queries on the tables in your Spark cluster. If you don't have any experience with SQL, don't worry (you can take our Introduction to SQL course!), we'll provide you with queries!

As you saw in the last exercise, one of the tables in your cluster is the flights table. This table contains a row for every flight that left Portland International Airport (PDX) or Seattle-Tacoma International Airport (SEA) in 2014 and 2015.

Running a query on this table is as easy as using the .sql() method on your SparkSession. This method takes a string containing the query and returns a DataFrame with the results!

If you look closely, you'll notice that the table flights is only mentioned in the query, not as an argument to any of the methods. This is because there isn't a local object in your environment that holds that data, so it wouldn't make sense to pass the table as an argument.

Remember, we've already created a SparkSession called spark in your workspace.
Instructions
100xp

    Use the .sql() method to get the first 10 rows of the flights table and save the result to flights10. The variable query contains the appropriate SQL query.
    Use the DataFrame method .show() to print flights10.

    Take Hint (-30xp)
'''
# Don't change this query
query = "FROM flights SELECT * LIMIT 10"

# Get the first 10 rows of flights
flights10 = ____

# Show the results
flights10.____


'''
Pandafy a Spark DataFrame

Suppose you've run a query on your huge dataset and aggregated it down to something a little more manageable.

Sometimes it makes sense to then take that table and work with it locally using a tool like pandas. Spark DataFrames make that easy with the .toPandas() method. Calling this method on a Spark DataFrame returns the corresponding pandas DataFrame. It's as simple as that!

This time the query counts the number of flights to each airport from SEA and PDX.

Remember, there's already a SparkSession called spark in your workspace!
Instructions
100xp

    Run the query using the .sql() method. Save the result in flight_counts.
    Use the .toPandas() method on flight_counts to create a pandas DataFrame called pd_counts.
    Print the .head() of pd_counts to the console.

    Take Hint (-30xp)
'''
# Don't change this query
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"

# Run the query
flight_counts = ____

# Convert the results to a pandas DataFrame
pd_counts = _____

# Print the head of pd_counts
print(____)


'''
Put some Spark in your data

In the last exercise, you saw how to move data from Spark to pandas. However, maybe you want to go the other direction, and put a pandas DataFrame into a Spark cluster! The SparkSession class has a method for this as well.

The .createDataFrame() method takes a pandas DataFrame and returns a Spark DataFrame.

The output of this method is stored locally, not in the SparkSession catalog. This means that you can use all the Spark DataFrame methods on it, but you can't access the data in other contexts.

For example, a SQL query (using the .sql() method) that references your DataFrame will throw an error. To access the data in this way, you have to save it as a temporary table.

You can do this using the .createTempView() Spark DataFrame method, which takes as its only argument the name of the temporary table you'd like to register. This method registers the DataFrame as a table in the catalog, but as this table is temporary, it can only be accessed from the specific SparkSession used to create the Spark DataFrame.

There is also the method .createOrReplaceTempView(). This safely creates a new temporary table if nothing was there before, or updates an existing table if one was already defined. You'll use this method to avoid running into problems with duplicate tables.

Check out the diagram to see all the different ways your Spark data structures interact with each other.

There's already a SparkSession called spark in your workspace, numpy has been imported as np, and pandas as pd.
Instructions
100xp
Instructions
100xp

    The code to create a pandas DataFrame of random numbers has already been provided and saved under pd_temp.
    Create a Spark DataFrame called spark_temp by calling the .createDataFrame() method with pd_temp as the argument.
    Examine the list of tables in your Spark cluster and verify that the new DataFrame is not present. Remember you can use spark.catalog.listTables() to do so.
    Register spark_temp as a temporary table named "temp" using the .createOrReplaceTempView() method. Rememeber that the table name is set including it as the only argument!
    Examine the list of tables again!

    Take Hint (-30xp)
'''
# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))

# Create spark_temp from pd_temp
spark_temp = _____

# Examine the tables in the catalog
print(____)

# Add spark_temp to the catalog
spark_temp.____

# Examine the tables in the catalog again
print(____)


'''
Dropping the middle man

Now you know how to put data into Spark via pandas, but you're probably wondering why deal with pandas at all? Wouldn't it be easier to just read a text file straight into Spark? Of course it would!

Luckily, your SparkSession has a .read attribute which has several methods for reading different data sources into Spark DataFrames. Using these you can create a DataFrame from a .csv file just like with regular pandas DataFrames!

The variable file_path is a string with the path to the file airports.csv. This file contains information about different airports all over the world.

A SparkSession named spark is available in your workspace.
Instructions
100xp

    Use the .read.csv() method to create a Spark DataFrame called airports
        The first argument is file_path
        Pass the argument header=True so that Spark knows to take the column names from the first line of the file.
    Print out this DataFrame by calling .show().

    Take Hint (-30xp)
'''
# Don't change this file path
file_path = "/usr/local/share/datasets/airports.csv"

# Read in the airports data
airports = _____

# Show the data



'''
Creating columns

In this chapter, you'll learn how to use the methods defined by Spark's DataFrame class to perform common data operations.

Let's look at performing column-wise operations. In Spark you can do this using the .withColumn() method, which takes two arguments. First, a string with the name of your new column, and second the new column itself.

The new column must be an object of class Column. Creating one of these is as easy as extracting a column from your DataFrame using df.colName.

Updating a Spark DataFrame is somewhat different than working in pandas because the Spark DataFrame is immutable. This means that it can't be changed, and so columns can't be updated in place.

Thus, all these methods return a new DataFrame. To overwrite the original DataFrame you must reassign the returned DataFrame using the method like so:

df = df.withColumn("newCol", df.oldCol + 1)

The above code creates a DataFrame with the same columns as df plus a new column, newCol, where every entry is equal to the corresponding entry from oldCol, plus one.

To overwrite an existing column, just pass the name of the column as the first argument!

Remember, a SparkSession called spark is already in your workspace.
Instructions
100xp

    Use the spark.table() method with the argument "flights" to create a DataFrame containing the values of the flights table in the .catalog. Save it as flights.
    Print the output of flights.show(). The column air_time contains the duration of the flight in minutes.
    Update flights to include a new column called duration_hrs, that contains the duration of each flight in hours.

    Take Hint (-30xp)
'''
# Create the DataFrame flights
flights = spark.table(____)

# Show the head
print(____)

# Add duration_hrs
flights = flights.withColumn(____)

'''
Filtering Data

Now that you have a bit of SQL know-how under your belt, it's easier to talk about the analogous operations using Spark DataFrames.

Let's take a look at the .filter() method. As you might suspect, this is the Spark counterpart of SQL's WHERE clause. The .filter() method takes either a Spark Column of boolean (True/False) values or the WHERE clause of a SQL expression as a string.

For example, the following two expressions will produce the same output:

flights.filter(flights.air_time > 120).show()
flights.filter("air_time > 120").show()

Remember, a SparkSession called spark is already in your workspace, along with the Spark DataFrame flights.
Instructions
100xp

    Use the .filter() method to find all the flights that flew over 1000 miles two ways:
        First, pass a SQL string to .filter() that checks the distance is greater than 1000. Save this as long_flights1.
        Then pass a boolean column to .filter() that checks the same thing. Save this as long_flights2.
    Print the .show() of both DataFrames and make sure they're actually equal!

    Take Hint (-30xp)
'''
# Filter flights with a SQL string
long_flights1 = ____

# Filter flights with a boolean column
long_flights2 = ____

# Examine the data to check they're equal
print(____)
print(____)


'''
Selecting

The Spark variant of SQL's SELECT is the .select() method. This method takes multiple arguments - one for each column you want to select. These arguments can either be the column name as a string (one for each column) or a column object (using the df.colName syntax). When you pass a column object, you can perform operations like addition or subtraction on the column to change the data contained in it, much like inside .withColumn().

The difference between .select() and .withColumn() methods is that .select() returns only the columns you specify, while .withColumn() returns all the columns of the DataFrame in addition to the one you defined. It's often a good idea to drop columns you don't need at the beginning of an operation so that you're not dragging around extra data as you're wrangling. In this case, you would use .select() and not .withColumn().

Remember, a SparkSession called spark is already in your workspace, along with the Spark DataFrame flights.
Instructions
100xp

    Select the columns tailnum, origin, and dest from flights by passing the column names as strings. Save this as selected1.
    Select the columns origin, dest, and carrier using the df.colName syntax and then filter the result using both of the filters already defined for you (filterA and filterB) to only keep flights from SEA to PDX. Save this as selected2.

    Take Hint (-30xp)
'''
# Select the first set of columns
selected1 = flights.select(____)

# Select the second set of columns
temp = flights.select(___, ___, ___)

# Define first filter
filterA = flights.origin == "SEA"

# Define second filter
filterB = flights.dest == "PDX"

# Filter the data, first by filterA then by filterB
selected2 = temp.filter(___).filter(___)


'''
Selecting II

Similar to SQL, you can also use the .select() method to perform column-wise operations. When you're selecting a column using the df.colName notation, you can perform any column operation and the .select() method will return the transformed column. For example,

flights.select(flights.air_time/60)

returns a column of flight durations in hours instead of minutes. You can also use the .alias() method to rename a column you're selecting. So if you wanted to .select() the column duration_hrs (which isn't in your DataFrame) you could do

flights.select((flights.air_time/60).alias("duration_hrs"))

The equivalent Spark DataFrame method .selectExpr() takes SQL expressions as a string:

flights.selectExpr("air_time/60 as duration_hrs")

with the SQL as keyword being equivalent to the .alias() method. To select multiple columns, you can pass multiple strings.

Remember, a SparkSession called spark is already in your workspace, along with the Spark DataFrame flights.
Instructions
100xp

Create a table of the average speed of each flight both ways.

    Calculate average speed by dividing the distance by the air_time (converted to hours). Use the .alias() method name this column "avg_speed". Save the output as the variable avg_speed.
    Select the columns "origin", "dest", "tailnum", and avg_speed (without quotes!). Save this as speed1.
    Create the same table using .selectExpr() and a string containing a SQL expression. Save this as speed2.

    Take Hint (-30xp)
'''
# Define avg_speed
avg_speed = (flights.___/(flights.___/60)).alias("___")

# Select the correct columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression
speed2 = flights.selectExpr("___", "___", "___", "distance/(air_time/60) as ___")


'''
Aggregating

All of the common aggregation methods, like .min(), .max(), and .count() are GroupedData methods. These are created by calling the .groupBy() DataFrame method. You'll learn exactly what that means in a few exercises. For now, all you have to do to use these functions is call that method on your DataFrame. For example, to find the minimum value of a column, col, in a DataFrame, df, you could do

df.groupBy().min("col").show()

This creates a GroupedData object (so you can use the .min() method), then finds the minimum value in col, and returns it as a DataFrame.

Now you're ready to do some aggregating of your own!

A SparkSession called spark is already in your workspace, along with the Spark DataFrame flights.
Instructions
100xp

    Find the length of the shortest (in terms of distance) flight that left PDX by first .filter()ing and using the .min() method.
    Find the length of the longest (in terms of time) flight that left SEA by filter()ing and using the .max() method.

    Take Hint (-30xp)
'''
# Find the shortest flight from PDX in terms of distance
flights.filter(____).groupBy().____.show()

# Find the longest flight from SEA in terms of duration
flights.filter(____).groupBy().____.show()



'''
Aggregating II

To get you familiar with more of the built in aggregation methods, here's a few more exercises involving the flights table!

Remember, a SparkSession called spark is already in your workspace, along with the Spark DataFrame flights.
Instructions
100xp

    Use the .avg() method to get the average air time of Delta Airlines flights (where the carrier column has the value "DL") that left SEA. Th place of departure is stored in the column origin. show() the result.
    Use the .sum() method to get the total number of hours all planes in this dataset spent in the air by creating a column called duration_hrs from the column air_time. show() the result.

    Take Hint (-30xp)
'''
# Average duration of Delta flights
flights.filter(___).filter(___).groupBy().avg(___).show()

# Total hours in the air
flights.withColumn("___", flights.air_time/60).groupBy().sum(___).show()


'''
Grouping and Aggregating I

Part of what makes aggregating so powerful is the addition of groups. PySpark has a whole class devoted to grouped data frames: pyspark.sql.GroupedData, which you saw in the last two exercises.

You've learned how to create a grouped DataFrame by calling the .groupBy() method on a DataFrame with no arguments.

Now you'll see that when you pass the name of one or more columns in your DataFrame to the .groupBy() method, the aggregation methods behave like when you use a GROUP BY statement in a SQL query!

Remember, a SparkSession called spark is already in your workspace, along with the Spark DataFrame flights.
Instructions
100xp

    Create a DataFrame called by_plane that is grouped by the column tailnum.
    Use the .count() method with no arguments to count the number of flights each plane made.
    Create a DataFrame called by_origin that is grouped by the column origin.
    Find the .avg() of the air_time column to find average duration of flights from PDX and SEA.

    Take Hint (-30xp)
'''
# Group by tailnum
by_plane = flights.groupBy("____")

# Number of flights each plane made
by_plane.____.show()

# Group by origin
by_origin = flights.groupBy("____")

# Average duration of flights from PDX and SEA
by_origin.avg("____").show()



'''
Grouping and Aggregating II

In addition to the GroupedData methods you've already seen, there is also the .agg() method. This method lets you pass an aggregate column expression that uses any of the aggregate functions from the pyspark.sql.functions submodule.

This submodule contains many useful functions for computing things like standard deviations. All the aggregation functions in this submodule take the name of a column in a GroupedData table.

Remember, a SparkSession called spark is already in your workspace, along with the Spark DataFrame flights. The grouped DataFrames you created in the last exercise are also in your workspace.
Instructions
100xp

    Import the submodule pyspark.sql.functions as F.
    Create a GroupedData table called by_month_dest that's grouped by both the month and dest columns.
    Use the .avg() method on the by_month_dest DataFrame to get the average dep_delay in each month for each destination.
    Find the corresponding standard deviation of each average by using the .agg() method with the function F.stddev().

    Take Hint (-30xp)
'''
# Import pyspark.sql.functions as F
import ____ as F

# Group by month and dest
by_month_dest = flights.groupBy(____)

# Average departure delay by month and destination
by_month_dest.____.show()

# Standard deviation
by_month_dest.agg(F.____(_____)).show()


'''
Joining II

In PySpark, joins are performed using the DataFrame method .join(). This method takes three arguments. The first is the second DataFrame that you want to join with the first one. The second argument, on, is the name of the key column(s) as a string. The names of the key column(s) must be the same in each table. The third argument, how, specifies the kind of join to perform. In this course we'll always use the value how="leftouter".

The flights dataset and a new dataset called airports are already in your workspace.
Instructions
100xp
Instructions
100xp

    Examine the airports DataFrame by printing the .show(). Note which key column will let you join airports to the flights table.
    Rename the faa column in airports to dest by re-assigning the result of airports.withColumnRenamed("faa", "dest") to airports.
    Join the airports DataFrame to the flights DataFrame on the dest column by calling the .join() method on flights. Save the result as flights_with_airports.
        The first argument should be the other DataFrame, airports.
        The second argument, on should be the key column.
        The third argument should be how="leftouter"
    Print the .show() of flights_with_airports. Note the new information that has been added.

    Take Hint (-30xp)
'''
# Examine the data
print(____)

# Rename the faa column
airports = ____

# Join the DataFrames
flights_with_airports = ____

# Examine the data again
print(____)


'''
DataCamp
Course Outline
5+
Machine Learning Pipelines

In the next two chapters you'll step through every stage of the machine learning pipeline, from data intake to model evaluation. Let's get to it!

At the core of the pyspark.ml module are the Transformer and Estimator classes. Almost every other class in the module behaves similarly to these two basic classes.

Transformer classes have a .transform() method that takes a DataFrame and returns a new DataFrame; usually the original one with a new column appended. For example, you might use the class Bucketizer to create discrete bins from a continuous feature or the class PCA to reduce the dimensionality of your dataset using principal component analysis.

Estimator classes all implement a .fit() method. These methods also take a DataFrame, but instead of returning another DataFrame they return a model object. This can be something like a StringIndexerModel for including categorical data saved as strings in your models, or a RandomForestModel that uses the random forest algorithm for classification or regression.

Which of the following is not true about machine learning in Spark?
Answer the question
50xp
Possible Answers

    Spark's algorithms give better results than other algorithms.
    press 1
    Working in Spark allows you to create reproducible machine learning pipelines.
    press 2
    Machine learning pipelines in Spark are made up of Transformers and Estimators.
    press 3
    PySpark uses the pyspark.ml submodule to interface with Spark's machine learning routines.
    press 4

    Take Hint (-15xp)


'''
'''
Join the DataFrames

In the next two chapters you'll be working to build a model that predicts whether or not a flight will be delayed based on the flights data we've been working with. This model will also include information about the plane that flew the route, so the first step is to join the two tables: flights and planes!
Instructions
100xp

    First, rename the year column of planes to plane_year to avoid duplicate column names.
    Create a new DataFrame called model_data by joining the flights table with planes using the tailnum column as the key.

    Take Hint (-30xp)
'''
# Rename year column
planes = planes.withColumnRenamed(____)

# Join the DataFrames
model_data = flights.join(____, on=____, how="leftouter")


'''
String to integer

Now you'll use the .cast() method you learned in the previous exercise to convert all the appropriate columns from your DataFrame model_data to integers!

As a little trick to make your code more readable, you can break long chains across multiple lines by putting .\ then calling the method on the next line, like so:

string = string.\
upper().\
lower()

The methods will be called from top to bottom.
Instructions
100xp

    Use the method .withColumn() to .cast() the following columns to type "integer".
        model_data.arr_delay
        model_data.air_time
        model_data.month
        model_data.plane_year

    Take Hint (-30xp)
'''
# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", ____)
model_data = model_data.withColumn("air_time", ____)
model_data = model_data.withColumn("month", ____)
model_data = model_data.withColumn("plane_year", ____)


'''
Create a new column

In the last exercise, you converted the column plane_year to an integer. This column holds the year each plane was manufactured. However, your model will use the planes' age, which is slightly different from the year it was made!
Instructions
100xp

    Create the column plane_age using the .withColumn() method and subtracting the year of manufacture (column plane_year) from the year (column year) of the flight.

    Take Hint (-30xp)
'''
# Create the column plane_age
model_data = model_data.withColumn("plane_age", ____)


'''
Making a Boolean

Consider that you're modeling a yes or no question: is the flight late? However, your data contains the arrival delay in minutes for each flight. Thus, you'll need to create a boolean column which indicates whether the flight was late or not!
Instructions
100xp

    Use the .withColumn() method to create the column is_late. This column is equal to model_data.arr_delay > 0.
    Convert this column to an integer column so that you can use it in your model and name it label (this is the default name for the response variable in Spark's machine learning routines).
    Filter out missing values (this has been done for you).

    Take Hint (-30xp)
'''
# Create is_late
model_data = model_data.withColumn("is_late", ____)

# Convert to an integer
model_data = model_data.withColumn("label", ____)

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")




'''
Carrier

In this exercise you'll create a StringIndexer and a OneHotEncoder to code the carrier column. To do this, you'll call the class constructors with the arguments inputCol and outputCol.

The inputCol is the name of the column you want to index or encode, and the outputCol is the name of the new column that the Transformer should create.
Instructions
100xp

    Create a StringIndexer called carr_indexer by calling StringIndexer() with inputCol="carrier" and outputCol="carrier_index".
    Create a OneHotEncoder called carr_encoder by calling OneHotEncoder() with inputCol="carrier_index" and outputCol="carrier_fact".

    Take Hint (-30xp)
'''
# Create a StringIndexer
carr_indexer = StringIndexer(____)

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(____)


'''
Destination

Now you'll encode the dest column just like you did in the previous exercise.
Instructions
100xp

    Create a StringIndexer called dest_indexer by calling StringIndexer() with inputCol="dest" and outputCol="dest_index".
    Create a OneHotEncoder called dest_encoder by calling OneHotEncoder() with inputCol="dest_index" and outputCol="dest_fact".

    Take Hint (-30xp)
'''
# Create a StringIndexer
dest_indexer = ____

# Create a OneHotEncoder
dest_encoder = ____


'''
Assemble a vector

Good work so far!

The last step in the Pipeline is to combine all of the columns containing our features into a single column. You can do this by storing each of the values from a column as an entry in a vector. Then, from the model's point of view, every observation is a vector that contains all of the information about it and a label that tells the modeler what value that observation corresponds to. All of the Spark modeling routines expect the data to be in this form.

Because of this, the pyspark.ml.feature submodule contains a class called VectorAssembler. This Transformer takes all of the columns you specify and combines them into a new vector column.
Instructions
100xp

    Create a VectorAssembler by calling VectorAssembler() with the inputCols names as a list and the outputCol name "features".
        The list of columns should be ["month", "air_time", "carrier_fact", "dest_fact", "plane_age"].

    Take Hint (-30xp)
'''
# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=____, outputCol=____)



'''
Create the pipeline

You're finally ready to create a Pipeline!

Pipeline is a class in the pyspark.ml module that combines all the Estimators and Transformers that you've already created. This lets you reuse the same modeling process over and over again by wrapping it up in one simple object. Neat, right?
Instructions
100xp

    Import Pipeline from pyspark.ml.
    Call the Pipeline() constructor with the keyword argument stages to create a Pipeline called flights_pipe.
        stages should be a list holding all the stages you want your data to go through in the pipeline. Here this is just [dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler]

    Take Hint (-30xp)
'''
# Import Pipeline
from ____ import ____

# Make the pipeline
flights_pipe = Pipeline(stages=____)


'''
Transform the data

Hooray, now you're finally ready to pass your data through the Pipeline you created!
Instructions
100xp

    Create the DataFrame piped_data by calling the Pipeline methods .fit() and .transform() in a chain. Both of these methods take model_data as their only argument.

    Take Hint (-30xp)
'''
# Fit and transform the data
piped_data = flights_pipe.____.____


'''
Split the data

Now that you've done all your manipulations, the last step before modeling is to split the data!
Instructions
100xp

    Use the DataFrame method .randomSplit() to split model_data into two pieces, training with 60% of the data, and test with 40% of the data by passing the list [.6, .4] to the .randomSplit() method.

    Take Hint (-30xp)
'''
# Split the data into training and test sets
training, test = piped_data.randomSplit(____)


'''
Create the modeler

The Estimator you'll be using is a LogisticRegression from the pyspark.ml.classification submodule.
Instructions
100xp

    Import the LogisticRegression class from pyspark.ml.classification.
    Create a LogisticRegression called lr by calling LogisticRegression() with no arguments.

    Take Hint (-30xp)
'''
# Import LogisticRegression
from ____ import ____

# Create a LogisticRegression Estimator
lr = ____


'''
Create the evaluator

The first thing you need when doing cross validation for model selection is a way to compare different models. Luckily, the pyspark.ml.evaluation submodule has classes for evaluating different kinds of models. Your model is a binary classification model, so you'll be using the BinaryClassificationEvaluator from the pyspark.ml.evaluation module.

This evaluator calculates the area under the ROC. This is a metric that combines the two kinds of errors a binary classifier can make (false positives and false negatives) into a simple number. You'll learn more about this towards the end of the chapter!
Instructions
100xp

    Import the submodule pyspark.ml.evaluation as evals.
    Create evaluator by calling evals.BinaryClassificationEvaluator() with the argument metricName="areaUnderROC".

    Take Hint (-30xp)
'''
# Import the evaluation submodule
import ____ as evals

# Create a BinaryClassificationEvaluator
evaluator = ____

'''
Make a grid

Next, you need to create a grid of values to search over when looking for the optimal hyperparameters. The submodule pyspark.ml.tuning includes a class called ParamGridBuilder that does just that (maybe you're starting to notice a pattern here; PySpark has a submodule for just about everything!).

You'll need to use the .addGrid() and .build() methods to create a grid that you can use for cross validation. The .addGrid() method takes a model parameter (an attribute of the model Estimator, lr, that you created a few exercises ago) and a list of values that you want to try. The .build() method takes no arguments, it just returns the grid that you'll use later.
Instructions
100xp
Instructions
100xp

    Import the submodule pyspark.ml.tuning under the alias tune.
    Call the class constructor ParamGridBuilder() with no arguments. Save this as grid.
    Call the .addGrid() method on grid with lr.regParam as the first argument and np.arange(0, .1, .01) as the second argument. This second call is a function from the numpy module (imported as np) that creates a list of numbers from 0 to .1, incrementing by .01. Overwrite grid with the result.
    Update grid again by calling the .addGrid() method a second time create a grid for lr.elasticNetParam that includes only the values [0, 1].
    Call the .build() method on grid and overwrite it with the output.

    Take Hint (-30xp)
'''
# Import the tuning submodule
import ____ as ____

# Create the parameter grid
grid = tune.____

# Add the hyperparameter
grid = grid.addGrid(____, np.arange(0, .1, .01))
grid = grid.addGrid(____, ____)

# Build the grid
grid = grid.build()


'''
Make the validator

The submodule pyspark.ml.tuning also has a class called CrossValidator for performing cross validation. This Estimator takes the modeler you want to fit, the grid of hyperparameters you created, and the evaluator you want to use to compare your models.

The submodule pyspark.ml.tune has already been imported as tune. You'll create the CrossValidator by passing it the logistic regression Estimator lr, the parameter grid, and the evaluator you created in the previous exercises.
Instructions
100xp

    Create a CrossValidator by calling tune.CrossValidator() with the arguments:
        estimator=lr
        estimatorParamMaps=grid
        evaluator=evaluator
    Name this object cv.

    Take Hint (-30xp)
'''
# Create the CrossValidator
cv = tune.____(estimator=____,
               estimatorParamMaps=____,
               evaluator=____
               )



'''
Fit the model(s)

You're finally ready to fit the models and select the best one!

Unfortunately, cross validation is a very computationally intensive procedure. Fitting all the models would take too long on DataCamp.

To do this locally you would use the code

# Fit cross validation models
models = cv.fit(training)

# Extract the best model
best_lr = models.bestModel

Remember, the training data is called training and you're using lr to fit a logistic regression model. Cross validation selected the parameter values regParam=0 and elasticNetParam=0 as being the best. These are the default values, so you don't need to do anything else with lr before fitting the model.
Instructions
100xp

    Create best_lr by calling lr.fit() on the training data.
    Print best_lr to verify that it's an object of the LogisticRegressionModel class.

    Take Hint (-30xp)
'''
# Call lr.fit()
best_lr = ____

# Print best_lr
print(____)


'''
Evaluate the model

Remember the test data that you set aside waaaaaay back in chapter 3? It's finally time to test your model on it! You can use the same evaluator you made to fit the model.
Instructions
100xp

    Use your model to generate predictions by applying best_lr.transform() to the test data. Save this as test_results.
    Call evaluator.evaluate() on test_results to compute the AUC. Print the output.

    Take Hint (-30xp)
'''
# Use the model to predict the test set
test_results = best_lr.____(____)

# Evaluate the predictions
print(evaluator.evaluate(____))


'''

'''