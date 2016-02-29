cd $TRAVIS_BUILD_DIR

#echo -e "Host butler.fri.uni-lj.si\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
#scp -i travis/key.private -P 5722 -r doc/build/html/ travis@butler.fri.uni-lj.si:/home/travis/html
mkdir doc/orange3doc
cp -r `pwd`/doc/data-mining-library/build/html doc/orange3doc/data-mining-library
cp -r `pwd`/doc/development/build/html doc/orange3doc/development
cp -r `pwd`/doc/visual-programming/build/html doc/orange3doc/visual-programming
echo -e "Host orange.biolab.si\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
echo -e "\tUser uploaddocs\n\tIdentityFile $TRAVIS_BUILD_DIR/.travis/key.private\n" >> ~/.ssh/config
rsync -a --delete doc/orange3doc/ orange.biolab.si:/orange3doc/
