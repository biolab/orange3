sudo docker run -e 'ACCEPT_EULA=Y' -e 'SA_PASSWORD=YourStrong!Passw0rd' \
   -p 1433:1433 -d microsoft/mssql-server-linux:2017-latest

export PYMSSQL_BUILD_WITH_BUNDLED_FREETDS=1
pip install pymssql

export ORANGE_TEST_DB_URI="${ORANGE_TEST_DB_URI}|mssql://SA:YourStrong!Passw0rd@0.0.0.0:1433"