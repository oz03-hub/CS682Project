zip -r project.zip . -x "logs/*" \
    -x "cs682/data/imdb/*" \
    -x "cs682/data/amazon/*" \
    -x "cs682/data/yelp/*" \
    -x "*__pycache__*" \
    -x "*.ruff_cache*" \
    -x "*.vscode*" \
    -x "train.sh"
