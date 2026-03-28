cd cs682/data

echo "Uzipping IMDB"
tar xzf imdb.tar.gz

echo "Unzipping Yelp"
tar xzf yelp.tar.gz
mv yelp_review_full_csv yelp

echo "Unzipping Amazon"
tar xzf amazon.tar.gz
mv amazon_review_full_csv amazon
