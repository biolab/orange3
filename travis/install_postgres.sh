VERSION=9.5alpha1
POSTGRES=$TRAVIS_BUILD_DIR/postgres/$VERSION

if [ ! "$(ls $POSTGRES)" ]; then
    mkdir -p $POSTGRES
    cd $POSTGRES

    # Download PostgreSQL and extension sources
    wget -O postgres.tar.bz2 https://ftp.postgresql.org/pub/source/v$VERSION/postgresql-$VERSION.tar.bz2
    tar xjf postgres.tar.bz2 --strip-components=1

    wget http://api.pgxn.org/dist/quantile/1.1.4/quantile-1.1.4.zip
    unzip quantile-1.1.4.zip

    wget https://dl.dropboxusercontent.com/u/3818654/binning.tar.gz
    tar xzf binning.tar.gz

    # Build and install PostgreSQL
    cd $POSTGRES
    ./configure --prefix $POSTGRES
    make install

    # Add our PostgreSQL to PATH, so extensions know where to install.
    export PATH=$POSTGRES/bin:$PATH

    # Install quantile extension
    cd $POSTGRES/quantile-1.1.4
    make install

    # Install binning extension
    cd $POSTGRES/binning
    make install
else
    echo "Using cached PostgreSQL."
fi

# Create a new database dir, create database test and register extensions
$POSTGRES/bin/initdb -D $TRAVIS_BUILD_DIR/db
$POSTGRES/bin/postgres -D $TRAVIS_BUILD_DIR/db -p 12345 &
sleep 1
$POSTGRES/bin/createdb -p 12345 test
$POSTGRES/bin/psql test -c 'CREATE EXTENSION quantile;' -p 12345
$POSTGRES/bin/psql test -c 'CREATE EXTENSION binning;' -p 12345

pip install psycopg2
export ORANGE_TEST_DB_URI=postgres://localhost:12345/test
