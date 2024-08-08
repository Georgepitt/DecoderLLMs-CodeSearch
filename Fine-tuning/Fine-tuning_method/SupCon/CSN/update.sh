#!/bin/bash

set -x

echo "Installing llm2vec..."
pip install llm2vec

if [ $? -ne 0 ]; then
    echo "Failed to install llm2vec"
    exit 1
fi

echo "Locating llm2vec package path..."
PACKAGE_PATH=$(python -c "import llm2vec; import os; print(os.path.join(os.path.dirname(llm2vec.__file__), 'dataset'))")

echo "llm2vec package path: $PACKAGE_PATH"

if [ ! -d "$PACKAGE_PATH" ]; then
    echo "Directory $PACKAGE_PATH does not exist"
    exit 1
fi

FILES=("CSNData.py" "__init__.py" "utils.py")

for FILE in "${FILES[@]}"; do
    if [ -f "$FILE" ]; then
        echo "Copying $FILE to $PACKAGE_PATH/"
        cp "$FILE" "$PACKAGE_PATH/"
        if [ $? -ne 0 ]; then
            echo "Failed to copy $FILE"
            exit 1
        fi
    else
        echo "File $FILE not found!"
    fi
done

echo "Successful update"
