import warnings
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime
import MySQLdb
import time

# Establish database connection
try:
    db = MySQLdb.connect(host="localhost", user="root", passwd="")
    cur = db.cursor()
except MySQLdb.OperationalError as e:
    print(f"Error: {e}")
    exit(1)

def ignore_warnings():
    """Function to ignore warnings."""
    warnings.warn("deprecated", DeprecationWarning)

def create_database_and_table():
    """Create database 'dharmapuri_db' and table 'power_prediction' if they do not exist."""
    cur.execute("CREATE DATABASE IF NOT EXISTS dharmapuri_db")
    cur.execute("USE dharmapuri_db")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS power_prediction (
            id INT AUTO_INCREMENT PRIMARY KEY,
            time_updated DATETIME,
            Temperature FLOAT,
            GHI FLOAT,
            DNI FLOAT,
            DHI FLOAT,
            power FLOAT
        )
    """)

def prepare_data(file_path):
    """Load and prepare data for training and prediction."""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :5].values
    Y_temp = df.iloc[:, 5].values
    Y_ghi = df.iloc[:, 6].values
    Y_dni = df.iloc[:, 7].values
    Y_dhi = df.iloc[:, 8].values
    
    return X, Y_temp, Y_ghi, Y_dni, Y_dhi

def train_models(X, Y_temp, Y_ghi, Y_dni, Y_dhi):
    """Train RandomForest models for temperature, GHI, DNI, and DHI predictions."""
    x_train, x_test, y_temp_train, y_temp_test = train_test_split(X, Y_temp, random_state=42)
    _, _, y_ghi_train, y_ghi_test = train_test_split(X, Y_ghi, random_state=42)
    _, _, y_dni_train, y_dni_test = train_test_split(X, Y_dni, random_state=42)
    _, _, y_dhi_train, y_dhi_test = train_test_split(X, Y_dhi, random_state=42)
    
    temp_model = RandomForestRegressor().fit(x_train, y_temp_train)
    ghi_model = RandomForestRegressor().fit(x_train, y_ghi_train)
    dni_model = RandomForestRegressor().fit(x_train, y_dni_train)
    dhi_model = RandomForestRegressor().fit(x_train, y_dhi_train)
    
    return temp_model, ghi_model, dni_model, dhi_model

def predict_future(temp_model, ghi_model, dni_model, dhi_model):
    """Predict future temperature, GHI, DNI, and DHI."""
    future_time = datetime.datetime.now() + datetime.timedelta(minutes=15)
    time_str = future_time.strftime("%Y-%m-%d %H:%M")
    time_values = list(map(int, time_str.replace('-', ' ').replace(':', ' ').split()))
    
    temp_prediction = temp_model.predict([time_values])[0]
    ghi_prediction = ghi_model.predict([time_values])[0]
    dni_prediction = dni_model.predict([time_values])[0]
    dhi_prediction = dhi_model.predict([time_values])[0]
    
    return time_str, temp_prediction, ghi_prediction, dni_prediction, dhi_prediction

def calculate_power(temp, ghi):
    """Calculate power based on predicted temperature and GHI."""
    efficiency = 0.18
    panel_area = 7.4322
    temp_adjustment = 1 - 0.05 * (temp - 25)
    
    power = efficiency * panel_area * ghi * temp_adjustment
    return power

def insert_into_database(cur, time_str, temp, ghi, dni, dhi, power):
    """Insert prediction results into the database."""
    sql = """
    INSERT INTO power_prediction (time_updated, Temperature, GHI, DNI, DHI, power)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        cur.execute(sql, (time_str, temp, ghi, dni, dhi, power))
        db.commit()
        print("Write complete")
    except Exception as e:
        db.rollback()
        print("Database error:", e)

def main():
    """Main function to run the prediction and storage process in a loop."""
    try:
        create_database_and_table()  # Create database and table if they do not exist
        
        X, Y_temp, Y_ghi, Y_dni, Y_dhi = prepare_data(r"D:\INTERNSHIP\power_dharmapuri.csv")
        temp_model, ghi_model, dni_model, dhi_model = train_models(X, Y_temp, Y_ghi, Y_dni, Y_dhi)

        while True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ignore_warnings()
            
            time_str, temp, ghi, dni, dhi = predict_future(temp_model, ghi_model, dni_model, dhi_model)
            power = calculate_power(temp, ghi)
            
            print(f"Time: {time_str}, Temperature: {temp:.2f}, GHI: {ghi:.2f}, DNI: {dni:.2f}, DHI: {dhi:.2f}, Power: {power:.2f}")
            
            insert_into_database(cur, time_str, temp, ghi, dni, dhi, power)
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("Process interrupted by user")
    finally:
        cur.close()
        db.close()

if __name__ == "__main__":
    main()
