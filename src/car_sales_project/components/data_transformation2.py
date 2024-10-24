import os
import pickle
from dataclasses import dataclass
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import logging

# Configure logging with file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "data_transformation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    raw_data_path: str = os.path.join(os.getcwd(), 'Data_Sets', r'C:\Users\SOOQ ELASER\car_sales_project\car_sales_data.csv')  
    transformed_train_data_path: str = os.path.join(os.getcwd(), 'artifact', 'train_transformed.pkl')
    transformed_test_data_path: str = os.path.join(os.getcwd(), 'artifact', 'test_transformed.pkl')

class DataTransformation:
    
    def __init__(self, path, threshold = 0.01):
        self.path = path
        self.target_column = 'Commission Earned'
        self.data_config = DataConfig()
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.smote = SMOTE()

        self.str_replace = [
            'Date', 'Salesperson', 'Customer Name', 'Car Make', 'Car Model',
            'Car Year', 'Sale Price', 'Commission Rate', 'Commission Earned'
        ]

    def read_data(self):
        """Reads the raw data and splits it for training or testing."""
        logger.info("Reading raw data from CSV file.")
        try:
            if not os.path.exists(self.data_config.raw_data_path):
                raise FileNotFoundError(f"File not found: {self.data_config.raw_data_path}")
        
            self.df = pd.read_csv(self.data_config.raw_data_path)
            test_size = 0.2
            random_state = 42
        
            # Split the data without stratification
            if 'test' in self.path:
                _, self.df = train_test_split(self.df, test_size=test_size, random_state=random_state)
            else:
                self.df, _ = train_test_split(self.df, test_size=test_size, random_state=random_state)
        
            logger.info("Data reading and splitting completed.")
            return self.df
        except Exception as e:
            logger.error(f"Error in reading data: {e}", exc_info=True)
            raise

    def replace_strings(self):
        """Removes specific prefixes from string columns and converts them to categorical codes."""
        try:
            for column in self.str_replace:
                if column in self.df.columns and self.df[column].dtype == 'object':
                    self.df[column] = self.df[column].str.replace(f"{column}_", "").astype('category').cat.codes
            logger.info("String replacements completed.")
        except Exception as e:
            logger.error(f"Error in replacing strings: {e}", exc_info=True)
            raise

    def one_hot_encode(self):
        """Applies one-hot encoding to the 'Car Make' and 'Car Model' columns."""
        try:
            encoder = OneHotEncoder(sparse_output=False)
            encoded = encoder.fit_transform(self.df[['Car Make', 'Car Model']])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Car Make', 'Car Model']))
            self.df = pd.concat([self.df.drop(['Car Make', 'Car Model'], axis=1), encoded_df], axis=1)
            logger.info("One-hot encoding completed.")
        except Exception as e:
            logger.error(f"Error in one-hot encoding: {e}", exc_info=True)
            raise

    def add_time_features(self):
        """Extracts date-time features from the 'Date' column."""
        try:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df['Year'] = self.df['Date'].dt.year
            self.df['Month'] = self.df['Date'].dt.month
            self.df['Day'] = self.df['Date'].dt.day
            logger.info("Time features extracted.")
        except Exception as e:
            logger.error(f"Error in adding time features: {e}", exc_info=True)
            raise

    def save_transformed_data(self):
        """Saves the transformed dataframe to the specified path."""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'wb') as f:
                pickle.dump(self.df, f)
            logger.info(f"Transformed data saved to {self.path}.")
        except Exception as e:
            logger.error(f"Error in saving transformed data: {e}", exc_info=True)
            raise

    def transform(self):
        """Executes the entire transformation pipeline."""
        try:
            logger.info("Starting data transformation process.")
            self.read_data()
            self.replace_strings()
            self.one_hot_encode()
            self.add_time_features()
            self.save_transformed_data()
            logger.info("Data transformation process completed successfully.")
        except Exception as e:
            logger.error(f"Error in the data transformation process: {e}", exc_info=True)
            raise

if __name__ == '__main__':
    # Initialize transformation objects
    train_transformer = DataTransformation(DataConfig().transformed_train_data_path)
    test_transformer = DataTransformation(DataConfig().transformed_test_data_path)
    
    # Execute transformations
    train_transformer.transform()
    test_transformer.transform()
