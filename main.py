import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfTransformer


class DataLoader:
    def __init__(self, product_data, review_data, product_columns, review_columns, key) -> None:
        self.product_data = product_data
        self.review_data = review_data
        self.product_columns = product_columns
        self.review_columns = review_columns
        self.key = key

    def load_data(self):
        products = pd.concat([pd.read_csv(file, sep='\t', names=self.product_columns) for file in self.product_data])
        reviews = pd.concat([pd.read_csv(file, sep='\t', names=self.review_columns) for file in self.review_data])

        return pd.merge(products, reviews, on = self.key, how='inner').drop_duplicates(subset=self.key)


class DataProcessor:
    def __init__(self, text_data, categorical_data:dict) -> None:
        self.text_data = text_data
        self.categorical_data = categorical_data

    def validate_categorical(self, loaded_data: pd.DataFrame ):
        for key, valid_categories in self.categorical_data.items():
            invalid_category = loaded_data[~loaded_data[key].isin(valid_categories)]
            if not invalid_category.empty:
                print("Invalid categories found:")
                print(invalid_category[key].unique())
    
    def correct(self, loaded_data: pd.DataFrame, column, from_cate, to_cate):
        loaded_data[column] = loaded_data[column].replace(from_cate, to_cate)


class ClassifierModel:
    def __init__(self, text_features, categorical_features, model) -> None:
        text_transformer = make_pipeline(
            CountVectorizer(),
            TfidfTransformer(),
        )
        feature_transformation = ColumnTransformer(
            transformers=[
                *[('feature_'+feature, text_transformer, feature ) for feature in text_features],
                ('categorical_features', OneHotEncoder(), categorical_features)
            ]
        )
        self.model = make_pipeline(
            feature_transformation,
            model
        )
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)

        print("accuracy:", accuracy)

        print(classification_report(y_test, y_pred))


def main():
    # configuration area
    products = ["./dataset/products-data-0.tsv", 
                "./dataset/products-data-1.tsv",
                "./dataset/products-data-2.tsv",
                "./dataset/products-data-3.tsv",]
    reviews = ["./dataset/reviews-0.tsv",
               "./dataset/reviews-1.tsv",
               "./dataset/reviews-2.tsv",
               "./dataset/reviews-3.tsv",]
    
    products_columns = ['id', 'category', 'product_title']
    reviews_columns = ['id', 'rating', 'review_text']
    key = 'id'
    
    text_features = ['product_title', 'review_text']
    categorical_features = ['rating']
    target_column = 'category'

    categorical_validation = {
        'rating' : [1, 2, 3, 4, 5],
        'category' : ['Kitchen', 'Jewelry']
    }
    random_seed = 123
    

    data_loader = DataLoader(products, reviews, products_columns, reviews_columns, key)
    data_processor = DataProcessor(text_features, categorical_validation)
    model = ClassifierModel(text_features, categorical_features, LogisticRegression(max_iter=1000))

    # load data
    data_loaded = data_loader.load_data()

    # process data
    data_processor.validate_categorical(data_loaded)
    data_processor.correct(data_loaded, target_column, 'Ktchen', 'Kitchen')

    X = data_loaded[text_features + categorical_features]
    y = data_loaded[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.train(X_train, y_train)

    y_pred = model.predict(X_test)

    model.evaluate(y_test, y_pred)

if __name__ == '__main__':
    main()
