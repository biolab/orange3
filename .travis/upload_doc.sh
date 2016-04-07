cd "$TRAVIS_BUILD_DIR"

# Decrypt private key
openssl aes-256-cbc -K $encrypted_3fc26dee5a84_key -iv $encrypted_3fc26dee5a84_iv -in .travis/upload_doc_id -out .travis/key.private -d
chmod 700 .travis/key.private

# Upload the docs
mkdir doc/orange3doc
cp -r doc/data-mining-library/build/html doc/orange3doc/data-mining-library
cp -r doc/development/build/html doc/orange3doc/development
cp -r doc/visual-programming/build/html doc/orange3doc/visual-programming
> ~/.ssh/config echo "
Host orange.biolab.si
    StrictHostKeyChecking no
    User uploaddocs
    IdentityFile $TRAVIS_BUILD_DIR/.travis/key.private
"
rsync -a --delete doc/orange3doc/ orange.biolab.si:/orange3doc/
